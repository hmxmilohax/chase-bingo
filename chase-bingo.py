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



def _draw_difficulty_badge(
    draw: ImageDraw.ImageDraw,
    top_left: Tuple[int, int],
    label: str,
    filled_count: int,
    style: str,  # 'normal' | 'impossible' | 'devil'
    fonts: Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont],
    base_img: Optional[Image.Image] = None,
    devil_img: Optional[Image.Image] = None
):
    (title_font, cell_font) = fonts
    x, y = top_left

    # --- layout constants ---
    BADGE_LABEL_GAP = 10     # space between label and pips row (more breathing room)
    circle_outer_r = 10
    circle_inner_r = 6
    spacing = 8

    # Draw the difficulty label
    bbox = draw.textbbox((0, 0), label, font=title_font)
    lh = bbox[3] - bbox[1]
    draw.text((x, y), label, font=title_font, fill='black')

    row_y = y + lh + BADGE_LABEL_GAP
    row_x = x

    # DEVIL: 5 devil icons instead of circles
    if style == 'devil' and devil_img is not None and base_img is not None:
        # scale devil to match the circle row height (≈ 20px)
        target_h = 2 * circle_outer_r  # 20px
        scale = target_h / max(1, devil_img.height)
        icon_w = int(devil_img.width * scale)
        icon_h = int(devil_img.height * scale)
        icon = devil_img.resize((icon_w, icon_h), Image.LANCZOS)

        # paste exactly 'filled_count' devils (normally 5)
        for i in range(filled_count):
            cx = row_x + i * ((2 * circle_outer_r) + spacing)
            base_img.paste(icon, (cx, row_y), icon)
        return

    # NORMAL / IMPOSSIBLE: circles with filled pips
    outline_color = 'black'
    outline_width = 2
    fill_color = (0, 0, 0)
    if style == 'impossible':
        fill_color = (200, 0, 0)  # red pips for Impossible

    # Outlines
    for i in range(5):
        cx = row_x + i * ((circle_outer_r * 2) + spacing) + circle_outer_r
        cy = row_y + circle_outer_r
        draw.ellipse(
            [cx - circle_outer_r, cy - circle_outer_r, cx + circle_outer_r, cy + circle_outer_r],
            outline=outline_color, width=outline_width
        )

    # Fills
    for i in range(filled_count):
        cx = row_x + i * ((circle_outer_r * 2) + spacing) + circle_outer_r
        cy = row_y + circle_outer_r
        draw.ellipse(
            [cx - circle_inner_r, cy - circle_inner_r, cx + circle_inner_r, cy + circle_inner_r],
            fill=fill_color
        )


def generate_bingo_image(spaces: list[str], username: str, generation: int = 1, difficulty_name: str = "Solid") -> bytes:
    # --- layout base ---
    margin = 20
    footer_height = 40
    board_size = 600  # base width for canvas calculations

    # width + cell sizes are stable
    width = board_size + 2 * margin
    cell = (width - 2 * margin) // 5
    board_draw_size = cell * 5

    # --- fonts first (we need them to measure header) ---
    try:
        title_font = ImageFont.truetype('arial.ttf', size=24)
        cell_font = ImageFont.truetype('arial.ttf', size=16)
        footer_font = ImageFont.truetype('arial.ttf', size=18)
    except IOError:
        title_font = ImageFont.load_default()
        cell_font = ImageFont.load_default()
        footer_font = ImageFont.load_default()

    # --- difficulty style + assets ---
    filled = FILLED_BY_NAME.get(difficulty_name, FILLED_BY_NAME['Solid'])
    style = 'devil' if difficulty_name == 'Devil' else ('impossible' if difficulty_name == 'Impossible' else 'normal')

    # measure label height with a temp drawer
    tmp = Image.new('RGBA', (10, 10), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    label_h = tmp_draw.textbbox((0, 0), difficulty_name, font=title_font)
    label_h = label_h[3] - label_h[1]

    # pips row height
    circle_row_h = 20  # 2 * 10 outer radius
    devil_img = None
    if style == 'devil':
        try:
            devil_img = Image.open(os.path.join(os.path.dirname(__file__), 'images', 'devil.png')).convert('RGBA')
            # we will scale to circle_row_h in the badge drawer
        except Exception:
            devil_img = None

    # header text and measurements
    now = datetime.now(timezone.utc)
    header_text = now.strftime(f"LA Chase Bingo - %B %d, %Y %H:%M:%S UTC (Gen {generation})")
    th_bbox = tmp_draw.textbbox((0, 0), header_text, font=title_font)
    title_h = th_bbox[3] - th_bbox[1]

    # spacing constants
    BADGE_TOP_PAD = 8
    BADGE_LABEL_GAP = 10
    HEADER_GAP_BELOW_PIPS = 12   # extra space under badge row before the title
    HEADER_GAP_BELOW_TITLE = 14  # extra space under the title before the board

    # compute header height dynamically so nothing overlaps
    badge_block_h = BADGE_TOP_PAD + label_h + BADGE_LABEL_GAP + circle_row_h
    header_height = max(
        90,
        badge_block_h + HEADER_GAP_BELOW_PIPS + title_h + HEADER_GAP_BELOW_TITLE
    )

    # now we can create the canvas using computed header height
    total_height = header_height + board_draw_size + footer_height
    img = Image.new('RGBA', (width, total_height), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # subtle watermark
    watermark_text = 'B I N G O'
    bbox = draw.textbbox((0, 0), watermark_text, font=title_font)
    wmw, wmh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    watermark = Image.new('RGBA', (width, total_height), (255, 255, 255, 0))
    wm_draw = ImageDraw.Draw(watermark)
    x_w = (width - wmw) // 2
    y_w = (total_height - wmh) // 2
    wm_draw.text((x_w, y_w), watermark_text, font=title_font, fill=(0, 0, 0, 40))
    wm_rot = watermark.rotate(-45, expand=1)
    img.paste(wm_rot, ((width - wm_rot.width) // 2, (total_height - wm_rot.height) // 2), wm_rot)

    # header title y-position (after badge block + gap)
    title_y = BADGE_TOP_PAD + label_h + BADGE_LABEL_GAP + circle_row_h + HEADER_GAP_BELOW_PIPS
    ht_bbox = draw.textbbox((0, 0), header_text, font=title_font)
    tw = ht_bbox[2] - ht_bbox[0]
    draw.text(((width - tw) / 2, title_y), header_text, font=title_font, fill='black')

    # draw the badge block (top-left corner)
    _draw_difficulty_badge(
        draw,
        (12, 8),                        # badge_padding_x, badge_padding_y
        difficulty_name,
        filled,
        style,
        (title_font, cell_font),
        base_img=img,
        devil_img=devil_img
    )

    # Prepare board data with center free space
    free_space_text = 'Takes place in Los Angeles FREE SPACE'
    if difficulty_name == 'Devil':
        # Use 25 actual prompts, no free space inserted
        pool = [s for s in spaces if s != free_space_text]
        if len(pool) >= 25:
            chosen = random.sample(pool, 25)
        else:
            # fallback (shouldn’t happen if load_bingo_spaces enforces 25)
            chosen = pool[:]
            while len(chosen) < 25 and pool:
                chosen.append(random.choice(pool))
    else:
        # Normal tiers keep the free space in the center
        available = [s for s in spaces if s != free_space_text]
        chosen = random.sample(available, 24)
        chosen.insert(12, free_space_text)

    # Draw cells
    start_y = header_height
    for idx, text in enumerate(chosen):
        row, col = divmod(idx, 5)
        x0 = margin + col * cell
        y0 = start_y + row * cell
        x1, y1 = x0 + cell, y0 + cell
        # soft‑grey checkerboard
        if (row + col) % 2 == 0:
            cell_bg = (245, 245, 245, 255)
        else:
            cell_bg = (220, 220, 220, 255)
        draw.rectangle([x0, y0, x1, y1], fill=cell_bg, outline='gray', width=1)
        if text.startswith("image/"):
            args = text.split("/")
            imagename = args[1]
            slotimg = Image.open("images/" + imagename + ".png")
            slotimg = slotimg.resize((cell, cell))
            img.paste(slotimg, [x0, y0, x1, y1], slotimg)
            continue

        # wrap text
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

        # compute line heights and spacing
        heights = [(draw.textbbox((0, 0), ln, font=cell_font)[3] - draw.textbbox((0, 0), ln, font=cell_font)[1]) for ln in lines]
        block_h = sum(heights) + (len(heights) - 1) * 4
        ty = y0 + (cell - block_h) / 2
        for ln, h in zip(lines, heights):
            bbox = draw.textbbox((0, 0), ln, font=cell_font)
            line_w = bbox[2] - bbox[0]
            tx = x0 + (cell - line_w) / 2
            draw.text((tx, ty), ln, font=cell_font, fill='black')
            ty += h + 4

    # Footer
    footer_text = f"{username}"
    bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
    fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    footer_y = header_height + board_draw_size + ((footer_height - fh) / 2)
    draw.text(((width - fw) / 2, footer_y), footer_text, font=footer_font, fill='black')

    # ─── animated-webp background overlay ───
    try:
        bg_path = os.path.join(os.path.dirname(__file__), 'images', 'bg.webp')
        webp = Image.open(bg_path)
        # extract all frames, convert and pick one at random
        frames = [f.copy().convert('RGBA') for f in ImageSequence.Iterator(webp)]
        bg = random.choice(frames)
        # force uniform 10% alpha
        bg.putalpha(int(255 * 0.1))
        bw, bh = bg.size
        # tile chosen frame to fill full canvas
        for y in range(0, total_height, bh):
            for x in range(0, width, bw):
                img.paste(bg, (x, y), bg)
    except Exception:
        # if missing or broken, just skip background
        pass

    # Save as PNG bytes
    final = img.convert('RGB')
    buf = io.BytesIO()
    final.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


if __name__ == '__main__':
    bot.run(discord_token)

