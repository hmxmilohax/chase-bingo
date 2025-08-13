import configparser
import os
import asyncio
import io
import random
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import discord
from discord.ext import commands, tasks
from discord import Object, app_commands
from bs4 import BeautifulSoup
import aiohttp
from typing import Optional, Tuple

# Load config
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

# Discord credentials & IDs
discord_token = config['discord']['token']
GUILD_ID = int(config['guild']['id'])
GUILD = Object(id=GUILD_ID)
ANNOUNCE_CHANNEL_ID = int(config['channels']['announce_channel_id'])
ROLE_ID = int(config['channels']['role_id'])
INACTIVE_CATEGORY_ID = int(config['channels']['inactive_category_id'])
ACTIVE_CATEGORY_ID = int(config['channels']['active_category_id'])
YOUTUBE_CHANNEL_ID = config['youtube']['channel_id']
TEST_MODE = False

# Bot setup
intents = discord.Intents.default()
intents.guilds = True
bot = commands.Bot(command_prefix=commands.when_mentioned, intents=intents)
first_ready = False

# State tracking
last_announced_video_id = None  # only announce when this changes
missing_misses = 0  # offline debounce counter
last_bingo_usage: dict[int, datetime] = {}
bingo_regenerated: dict[int, bool] = {}


# ── Difficulty ring atlas helpers ─────────────────────────────────────────────
RING_ATLAS_PATHS = [
    os.path.join(os.path.dirname(__file__), 'images', 'DifficultyRings.png'),
    os.path.join(os.path.dirname(__file__), 'DifficultyRings.png'),
    '/mnt/data/DifficultyRings.png',
]

RING_INDEX_BY_NAME = {
    "Warmup": 0,
    "Apprentice": 1,
    "Solid": 2,
    "Moderate": 3,
    "Challenging": 4,
    "Nightmare": 5,
    "Impossible": 6,
    "Devil": 7,
}

# Difficulty configuration
DIFFICULTY_ORDER = [
    ("Warmup", 1, 0),        # (display name, numeric alias, filled black circles)
    ("Apprentice", 2, 1),
    ("Solid", 3, 2),         # default
    ("Moderate", 4, 3),
    ("Challenging", 5, 4),
    ("Nightmare", 6, 5),
    ("Impossible", 7, 5),    # 5 filled, but red
    ("Devil", 7, 5),    # 5 filled, but red
]

# Weighted counts per difficulty (Easy/Medium/Hard sum to 25)
# Tune these to taste; “Solid” matches your current 20/3/2.
DIFFICULTY_WEIGHTS = {
    "Warmup":      (22, 2, 1),
    "Apprentice":  (21, 3, 1),
    "Solid":       (20, 3, 2),
    "Moderate":    (18, 5, 2),
    "Challenging": (16, 6, 3),
    "Nightmare":   (14, 7, 4),
    "Impossible":  (10, 8, 7),
    "Devil":       (0, 2, 23),
}
NAME_BY_NUMBER = {num: name for (name, num, _) in DIFFICULTY_ORDER}
FILLED_BY_NAME = {name: filled for (name, _, filled) in DIFFICULTY_ORDER}

async def safe_set_permissions(channel: discord.TextChannel, guild: discord.Guild, view: bool):
    if TEST_MODE:
        print("[TEST_MODE] Skipping safe_set_permissions()")
        return
    role = guild.get_role(ROLE_ID)
    if not channel.permissions_for(guild.me).manage_channels:
        return
    try:
        if view:
            await channel.set_permissions(guild.default_role, overwrite=None)
            await channel.set_permissions(role, view_channel=True)
        else:
            await channel.set_permissions(guild.default_role, view_channel=False)
            await channel.set_permissions(role, view_channel=False)
    except discord.Forbidden:
        print("[WARN] Missing permissions to set channel perms; skipping.")


def rel_ts(seconds: int) -> str:
    # returns a Discord dynamic timestamp for “seconds” in the future
    now_utc = datetime.now(timezone.utc)
    return f"<t:{int(now_utc.timestamp())+seconds}:R>"

async def find_channel(guild: discord.Guild, channel_id: int) -> discord.abc.GuildChannel:
    chan = bot.get_channel(channel_id)
    if chan:
        return chan
    try:
        return await bot.fetch_channel(channel_id)
    except (discord.Forbidden, Exception):
        return None

# --- initialize_channel: skip edits in TEST_MODE and catch Forbidden ---
async def initialize_channel():
    if not bot.guilds:
        return
    guild = bot.guilds[0]
    channel = await find_channel(guild, ANNOUNCE_CHANNEL_ID)
    role = guild.get_role(ROLE_ID)
    if not channel or not role:
        return

    if TEST_MODE:
        print("[TEST_MODE] Skipping channel move/permission changes in initialize_channel()")
        return

    inactive_cat = await find_channel(guild, INACTIVE_CATEGORY_ID)
    if isinstance(inactive_cat, discord.CategoryChannel):
        try:
            await channel.edit(category=inactive_cat)
        except discord.Forbidden:
            print("[WARN] Missing permissions to move channel; skipping.")

    perms = channel.permissions_for(guild.me)
    if perms.manage_channels and guild.me.top_role.position > role.position:
        try:
            await channel.set_permissions(guild.default_role, view_channel=False)
            await channel.set_permissions(role, view_channel=False)
        except discord.Forbidden:
            print("[WARN] Missing permissions to set channel perms; skipping.")
    else:
        print("[WARN] No manage_channels or role hierarchy too low; skipping perms.")


async def deferred_cleanup(channel, guild):
    await asyncio.sleep(600)
    await finish_cleanup(channel, guild)

@bot.event
async def on_ready():
    global first_ready
    if not first_ready:
        first_ready = True
        print(f'Logged in as {bot.user} (ID: {bot.user.id})')
        try:
            await initialize_channel()
        except Exception as e:
            print(f'[WARN] initialize_channel failed: {e!r}')
        # Avoid global sync; just do guild sync
        synced = await bot.tree.sync(guild=GUILD)
        print(f'Synced {len(synced)} guild commands')
        await check_live_status()
        check_live_status.start()
    else:
        print("on_ready fired again; skipping sync.")

@tasks.loop(seconds=60)
async def check_live_status():
    global last_announced_video_id, missing_misses
    if not bot.guilds:
        return
    guild = bot.guilds[0]
    channel = await find_channel(guild, ANNOUNCE_CHANNEL_ID)
    role = guild.get_role(ROLE_ID)
    if not channel or not role:
        return

    YOUTUBE_LIVE_URL = f'https://www.youtube.com/channel/{YOUTUBE_CHANNEL_ID}/live'

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(YOUTUBE_LIVE_URL, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                if resp.status != 200:
                    print(f'Failed to fetch YouTube live page: {resp.status}')
                    return
                html = await resp.text()
    except Exception as e:
        print(f'Error fetching YouTube live page: {e}')
        return

    soup = BeautifulSoup(html, 'html.parser')
    canonical = soup.find('link', rel='canonical')
    if not canonical:
        print('Canonical link not found')
        return

    href = canonical.get('href')
    if '/watch?v=' in href:
        vid = href.split('/watch?v=')[1]
        if vid != last_announced_video_id:
            missing_misses = 0
            link = f'https://youtu.be/{vid}'

            if not TEST_MODE:
                active_cat = await find_channel(guild, ACTIVE_CATEGORY_ID)
                if isinstance(active_cat, discord.CategoryChannel):
                    try:
                        await channel.edit(category=active_cat)
                    except discord.Forbidden:
                        print("[WARN] Missing permissions to move to active; skipping.")
                await safe_set_permissions(channel, guild, True)
            else:
                print("[TEST_MODE] Not moving channel or changing perms for live start.")

            prefix = '' if TEST_MODE else f'<@&{ROLE_ID}> '
            await channel.send(prefix + f'The chase is on: {link}\nUse `/bingo` to get your card!')
            last_announced_video_id = vid
    else:
        if last_announced_video_id is not None:
            missing_misses += 1
            if missing_misses >= 2:
                last_announced_video_id = None
                missing_misses = 0
                await channel.send(f"Stream is over — moving channel {rel_ts(600)}")
                if not TEST_MODE:
                    bot.loop.create_task(deferred_cleanup(channel, guild))
                else:
                    print("[TEST_MODE] Not scheduling deferred_cleanup().")



async def finish_cleanup(channel: discord.TextChannel, guild: discord.Guild):
    if TEST_MODE:
        print("[TEST_MODE] Skipping finish_cleanup() channel move/perm changes")
        return
    inactive_cat = await find_channel(guild, INACTIVE_CATEGORY_ID)
    if isinstance(inactive_cat, discord.CategoryChannel):
        await channel.edit(category=inactive_cat)
    await safe_set_permissions(channel, guild, False)
    await channel.send("Channel moved to inactive category and hidden.")


@bot.tree.command(guild=GUILD, name='bingo', description='Generate a bingo card for the current chase')
@app_commands.describe(difficulty="Pick a difficulty (optional)")
@app_commands.choices(
    difficulty=[
        # name choices
        app_commands.Choice(name="Warmup", value="Warmup"),
        app_commands.Choice(name="Apprentice", value="Apprentice"),
        app_commands.Choice(name="Solid", value="Solid"),
        app_commands.Choice(name="Moderate", value="Moderate"),
        app_commands.Choice(name="Challenging", value="Challenging"),
        app_commands.Choice(name="Nightmare", value="Nightmare"),
        app_commands.Choice(name="Impossible", value="Impossible"),
        app_commands.Choice(name="Devil", value="Devil"),
    ]
)
async def bingo(interaction: discord.Interaction, difficulty: Optional[app_commands.Choice[str]] = None):
    await interaction.response.defer(ephemeral=True)

    # ── regen window logic (unchanged) ──
    now = datetime.now(timezone.utc)
    uid = interaction.user.id
    first = last_bingo_usage.get(uid)
    regen = bingo_regenerated.get(uid, False)

    if first is None or (now - first).total_seconds() >= 3600:
        last_bingo_usage[uid] = now
        bingo_regenerated[uid] = False
        regen = False

    if first is not None and (now - first).total_seconds() < 3600 and regen:
        remaining = 3600 - (now - first).total_seconds()
        mins = int(remaining // 60) + 1
        return await interaction.followup.send(
            f'You’ve already regenerated once. Try again in {mins} minute(s).',
            ephemeral=True
        )

    is_regen = False
    if first is not None and (now - first).total_seconds() < 3600:
        bingo_regenerated[uid] = True
        is_regen = True

    # ── resolve difficulty from dropdown choice ──
    # (Explicit mapping so we don't depend on DIFFICULTY_ORDER/NAME_BY_NUMBER.)
    num_to_name = {
        1: "Warmup", 2: "Apprentice", 3: "Solid", 4: "Moderate",
        5: "Challenging", 6: "Nightmare", 7: "Impossible"
    }
    chosen_diff_name = "Solid"
    if difficulty:
        raw = difficulty.value.strip()
        if raw.isdigit():
            chosen_diff_name = num_to_name.get(int(raw), "Solid")
        else:
            chosen_diff_name = raw  # ← keep Devil as Devil
        if chosen_diff_name not in DIFFICULTY_WEIGHTS:
            chosen_diff_name = "Solid"

    # ── build spaces with chosen weights ──
    spaces = load_bingo_spaces(difficulty_name=chosen_diff_name)
    if len(spaces) < 25:
        return await interaction.followup.send('Not enough bingo spaces configured.', ephemeral=True)

    # ── render image ──
    gen = 2 if is_regen else 1
    image_bytes = generate_bingo_image(
        spaces,
        interaction.user.display_name,
        generation=gen,
        difficulty_name=chosen_diff_name
    )

    # ── DM first; fall back to ephemeral ──
    try:
        dm_file = discord.File(io.BytesIO(image_bytes), filename='bingo.png')
        await interaction.user.send("Here's your bingo board:", file=dm_file)
        response_text = 'I’ve DMed you your bingo card!'
        if is_regen:
            response_text += "\n\n⚠️ Please only play one board. You cannot regenerate again until the hour is up."
        await interaction.followup.send(response_text, ephemeral=True)
    except discord.Forbidden:
        fallback_file = discord.File(io.BytesIO(image_bytes), filename='bingo.png')
        warn = "\n\n⚠️ Please only play one board. You cannot regenerate again until the hour is up." if is_regen else ""
        await interaction.followup.send(
            f"I couldn't DM you (maybe your DMs are closed?). Here is your card:{warn}",
            file=fallback_file,
            ephemeral=True
        )


@bot.tree.command(guild=GUILD, name='chase', description='Get the current live chase link')
async def chase(interaction: discord.Interaction):
    if last_announced_video_id:
        await interaction.response.send_message(f'Live stream link: https://youtu.be/{last_announced_video_id}', ephemeral=True)
    else:
        await interaction.response.send_message('There is no live stream at the moment.', ephemeral=True)

def load_bingo_spaces(path: str = 'spaces.ini', difficulty_name: str = 'Solid') -> list[str]:
    spaces_cfg = configparser.ConfigParser(allow_no_value=True)
    spaces_cfg.optionxform = str
    spaces_cfg.read(os.path.join(os.path.dirname(__file__), path))

    easy_spaces = list(spaces_cfg['Easy'])
    medium_spaces = list(spaces_cfg['Medium'])
    hard_spaces = list(spaces_cfg['Hard'])

    e_cnt, m_cnt, h_cnt = DIFFICULTY_WEIGHTS.get(difficulty_name, DIFFICULTY_WEIGHTS['Solid'])

    selected = []
    selected += random.sample(easy_spaces, min(e_cnt, len(easy_spaces)))
    selected += random.sample(medium_spaces, min(m_cnt, len(medium_spaces)))
    selected += random.sample(hard_spaces, min(h_cnt, len(hard_spaces)))

    def fill_from_pool(pool: list[str]):
        nonlocal selected
        pool_left = [s for s in pool if s not in selected]
        random.shuffle(pool_left)
        while len(selected) < 25 and pool_left:
            selected.append(pool_left.pop())

    if difficulty_name == 'Devil':
        # Prefer Hard, then Medium, then Easy until we hit 25
        fill_from_pool(hard_spaces)
        fill_from_pool(medium_spaces)
        fill_from_pool(easy_spaces)
    else:
        # Original behavior
        all_spaces = easy_spaces + medium_spaces + hard_spaces
        remaining = list(set(all_spaces) - set(selected))
        while len(selected) < 25 and remaining:
            choice = random.choice(remaining)
            selected.append(choice)
            remaining.remove(choice)

    random.shuffle(selected)
    return selected

def _load_ring_atlas() -> Optional[Image.Image]:
    for p in RING_ATLAS_PATHS:
        if os.path.exists(p):
            try:
                return Image.open(p).convert('RGBA')
            except Exception:
                pass
    return None

def _get_ring_tile(atlas: Image.Image, diff_name: str) -> Image.Image:
    idx = RING_INDEX_BY_NAME.get(diff_name, 2)  # default to Solid
    tile = atlas.width // 3  # 128 for a 384 atlas
    r, c = divmod(idx, 3)
    box = (c * tile, r * tile, (c + 1) * tile, (r + 1) * tile)
    return atlas.crop(box)

def _draw_diff_ring_in_cell(
    base_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    cell_box: Tuple[int, int, int, int],
    diff_name: str,
    font: ImageFont.FreeTypeFont,
    atlas: Optional[Image.Image],
    overlay_text: Optional[str] = None
):
    x0, y0, x1, y1 = cell_box
    pad = 2
    gx0, gy0, gx1, gy1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad

    # cell bg
    draw.rectangle([gx0, gy0, gx1, gy1], fill=(204, 204, 204, 255))

    # ring placement
    side = max(1, min(gx1 - gx0, gy1 - gy0) - 4)
    rx = gx0 + ((gx1 - gx0) - side) // 2
    ry = gy0 + ((gy1 - gy0) - side) // 2
    if atlas:
        ring = _get_ring_tile(atlas, diff_name).resize((side, side), Image.LANCZOS)
        base_img.paste(ring, (rx, ry), ring)

    # text to draw
    text = overlay_text if overlay_text is not None else diff_name

    # wrap with explicit hard line breaks
    max_w = (gx1 - gx0) - 12
    lines = []
    for para in text.splitlines():          # ← preserves '\n'
        words = para.split()
        if not words:
            lines.append("")                # keep blank lines if any
            continue
        cur = words[0]
        for w in words[1:]:
            test = f"{cur} {w}"
            tb = draw.textbbox((0, 0), test, font=font)
            if (tb[2] - tb[0]) <= max_w:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)

    if len(lines) > 4:
        lines = lines[:4]
        lines[-1] += '…'

    # metrics
    line_gap = 4
    heights = [draw.textbbox((0, 0), ln, font=font)[3] - draw.textbbox((0, 0), ln, font=font)[1] for ln in lines]
    block_h = sum(heights) + (len(heights) - 1) * line_gap
    cx = (x0 + x1) // 2

    if diff_name == 'Devil':
        # icon under plate/text
        try:
            icon = Image.open(os.path.join(os.path.dirname(__file__), 'images', 'devil.png')).convert('RGBA')
            inner = max(1, int(side * 0.55))
            icon = icon.resize((inner, inner), Image.LANCZOS)
            ix = rx + (side - inner) // 2
            iy = ry + (side - inner) // 2
            base_img.paste(icon, (ix, iy), icon)
        except Exception:
            pass

        # metrics (lines, heights, block_h, cx already computed)
        max_line_w = 0
        for ln in lines:
            bb = draw.textbbox((0, 0), ln, font=font)
            max_line_w = max(max_line_w, bb[2] - bb[0])

        bottom_margin = 0                 # sit at the very bottom
        y_bottom = gy1 - 1                # 1px inset to avoid bleeding the cell border
        ty = y_bottom - block_h           # lines grow upward

        # tight padding + shave 1px header
        PAD_X = 6
        HEAD_TRIM = 1
        px0 = int(cx - max_line_w / 2) - PAD_X
        px1 = int(cx + max_line_w / 2) + PAD_X
        py0 = int(ty) - HEAD_TRIM
        py1 = int(y_bottom)

        # clamp
        px0 = max(px0, gx0 + 1)
        py0 = max(py0, gy0 + 1)
        px1 = min(px1, gx1 - 1)
        py1 = min(py1, gy1 - 1)

        # translucent plate (~30% opacity) via overlay, so alpha actually works
        plate_w = max(1, px1 - px0)
        plate_h = max(1, py1 - py0)
        plate = Image.new('RGBA', (plate_w, plate_h), (255, 255, 255, 140))  # 76 ≈ 30%
        base_img.paste(plate, (px0, py0), plate)

        # text over the plate (bottom-anchored block)
        TEXT_NUDGE = -3   # move text up 2px; try -1 if you want even subtler
        y_line = ty + TEXT_NUDGE
        for ln, h in zip(lines, heights):
            lb = draw.textbbox((0, 0), ln, font=font)
            lw = lb[2] - lb[0]
            draw.text((cx - lw / 2, y_line), ln, font=font, fill='black')
            y_line += h + line_gap
    else:
        # non-Devil: centered text (honors \n via splitlines() above)
        ty = y0 + ((y1 - y0) - block_h) // 2
        y_line = ty
        for ln, h in zip(lines, heights):
            lb = draw.textbbox((0, 0), ln, font=font)
            lw = lb[2] - lb[0]
            draw.text((cx - lw / 2, y_line), ln, font=font, fill='black')
            y_line += h + line_gap


def generate_bingo_image(spaces: list[str], username: str, generation: int = 1, difficulty_name: str = "Solid") -> bytes:
    margin = 20
    footer_height = 40
    board_size = 600
    width = board_size + 2 * margin
    cell = (width - 2 * margin) // 5
    board_draw_size = cell * 5

    try:
        title_font = ImageFont.truetype('arial.ttf', size=24)
        cell_font = ImageFont.truetype('arial.ttf', size=16)
        cell_font_small = ImageFont.truetype('arial.ttf', size=15)  # ← slightly smaller
        footer_font = ImageFont.truetype('arial.ttf', size=18)
    except IOError:
        title_font = ImageFont.load_default()
        cell_font = ImageFont.load_default()
        cell_font_small = cell_font  # fallback if TTF not available
        footer_font = ImageFont.load_default()

    # We only show a center title now (no corner dots/badge)
    now = datetime.now(timezone.utc)
    header_text = now.strftime(f"LA Chase Bingo - %B %d, %Y %H:%M:%S UTC (Gen {generation})")
    tmp = Image.new('RGBA', (10, 10), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    th_bbox = tmp_draw.textbbox((0, 0), header_text, font=title_font)
    title_h = th_bbox[3] - th_bbox[1]

    HEADER_GAP_BELOW_TITLE = 14
    header_height = max(70, title_h + 24 + HEADER_GAP_BELOW_TITLE)

    total_height = header_height + board_draw_size + footer_height
    img = Image.new('RGBA', (width, total_height), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # watermark
    watermark_text = 'B I N G O'
    wb = draw.textbbox((0, 0), watermark_text, font=title_font)
    wmw, wmh = wb[2] - wb[0], wb[3] - wb[1]
    watermark = Image.new('RGBA', (width, total_height), (255, 255, 255, 0))
    wm_draw = ImageDraw.Draw(watermark)
    x_w = (width - wmw) // 2
    y_w = (total_height - wmh) // 2
    wm_draw.text((x_w, y_w), watermark_text, font=title_font, fill=(0, 0, 0, 40))
    wm_rot = watermark.rotate(-45, expand=1)
    img.paste(wm_rot, ((width - wm_rot.width) // 2, (total_height - wm_rot.height) // 2), wm_rot)

    # center title
    ht_bbox = draw.textbbox((0, 0), header_text, font=title_font)
    tw = ht_bbox[2] - ht_bbox[0]
    draw.text(((width - tw) / 2, 12), header_text, font=title_font, fill='black')

    # ── choose spaces; center slot reserved for difficulty badge ──
    # ── board contents (no placeholder strings) ──
    free_space_text = 'Takes place in Los Angeles FREE SPACE'
    ring_atlas = _load_ring_atlas()
    is_devil = (difficulty_name.strip().lower() == 'devil')

    if is_devil:
        # Devil: 25 real prompts (no free space)
        pool = [s for s in spaces if s != free_space_text]
        if len(pool) >= 25:
            chosen = random.sample(pool, 25)
        else:
            chosen = pool[:]
            while len(chosen) < 25 and pool:
                chosen.append(random.choice(pool))
    else:
        # Other tiers: 24 prompts; center is the difficulty badge
        available = [s for s in spaces if s != free_space_text]
        chosen = random.sample(available, 24) if len(available) >= 24 else available[:]
        while len(chosen) < 24 and available:
            chosen.append(random.choice(available))
        # insert a dummy to keep indexing simple (we'll override draw at idx==12)
        chosen.insert(12, None)

    # ── draw cells ──
    start_y = header_height
    for idx, text in enumerate(chosen):
        row, col = divmod(idx, 5)
        x0 = margin + col * cell
        y0 = start_y + row * cell
        x1, y1 = x0 + cell, y0 + cell

        # background
        cell_bg = (245, 245, 245, 255) if (row + col) % 2 == 0 else (220, 220, 220, 255)
        draw.rectangle([x0, y0, x1, y1], fill=cell_bg, outline='gray', width=1)

        # center cell special handling
        if idx == 12:
            if is_devil:
                # show Devil ring + the prompt text INSIDE the ring
                center_text = text if isinstance(text, str) else ''
                _draw_diff_ring_in_cell(img, draw, (x0, y0, x1, y1),
                                        'Devil', cell_font, ring_atlas,
                                        overlay_text=center_text)
            else:
                # non-Devil: ring matches the difficulty; inside text marks it as the Free Space
                label = f"{difficulty_name}\n(Free Space)"
                _draw_diff_ring_in_cell(
                    img, draw, (x0, y0, x1, y1),
                    difficulty_name, cell_font_small, ring_atlas,  # ← use smaller font here
                    overlay_text=label
                )
            continue  # center already drawn


        # images-in-cells still supported
        if isinstance(text, str) and text.startswith("image/"):
            imagename = text.split("/")[1]
            slotimg = Image.open("images/" + imagename + ".png").resize((cell, cell))
            img.paste(slotimg, [x0, y0, x1, y1], slotimg)
            continue

        # normal prompt rendering
        if not isinstance(text, str):
            text = ''  # safety
        max_w = cell - 10
        words = text.split()
        lines, cur = [], ''
        for w in words:
            test = (cur + ' ' + w).strip()
            bbox = draw.textbbox((0, 0), test, font=cell_font)
            if bbox[2] - bbox[0] <= max_w:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        if len(lines) > 4:
            lines = lines[:4]
            lines[-1] += '…'

        heights = [(draw.textbbox((0, 0), ln, font=cell_font)[3] - draw.textbbox((0, 0), ln, font=cell_font)[1]) for ln in lines]
        block_h = sum(heights) + (len(heights) - 1) * 4
        ty = y0 + (cell - block_h) / 2
        for ln, h in zip(lines, heights):
            bbox = draw.textbbox((0, 0), ln, font=cell_font)
            line_w = bbox[2] - bbox[0]
            tx = x0 + (cell - line_w) / 2
            draw.text((tx, ty), ln, font=cell_font, fill='black')
            ty += h + 4

    # footer
    footer_text = f"{username}"
    bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
    fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    footer_y = header_height + board_draw_size + ((footer_height - fh) / 2)
    draw.text(((width - fw) / 2, footer_y), footer_text, font=footer_font, fill='black')

    # optional animated-webp bg
    try:
        bg_path = os.path.join(os.path.dirname(__file__), 'images', 'bg.webp')
        webp = Image.open(bg_path)
        frames = [f.copy().convert('RGBA') for f in ImageSequence.Iterator(webp)]
        bg = random.choice(frames)
        bg.putalpha(int(255 * 0.1))
        bw, bh = bg.size
        for y in range(0, total_height, bh):
            for x in range(0, width, bw):
                img.paste(bg, (x, y), bg)
    except Exception:
        pass

    final = img.convert('RGB')
    buf = io.BytesIO()
    final.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()



if __name__ == '__main__':
    bot.run(discord_token)

