import configparser
import os
import asyncio
import io
import random
import datetime
from PIL import Image, ImageDraw, ImageFont
import discord
from discord.ext import commands, tasks
from discord import Object
from bs4 import BeautifulSoup
import aiohttp

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
last_bingo_usage: dict[int, datetime.datetime] = {}
bingo_regenerated: dict[int, bool] = {}

def rel_ts(seconds: int) -> str:
    # returns a Discord dynamic timestamp for “seconds” in the future
    run_at = datetime.datetime.utcnow() + datetime.timedelta(seconds=seconds)
    return f"<t:{int(run_at.timestamp())}:R>"

async def find_channel(guild: discord.Guild, channel_id: int) -> discord.abc.GuildChannel:
    chan = bot.get_channel(channel_id)
    if chan:
        return chan
    try:
        return await bot.fetch_channel(channel_id)
    except (discord.Forbidden, Exception):
        return None

async def initialize_channel():
    if not bot.guilds:
        return
    guild = bot.guilds[0]
    channel = await find_channel(guild, ANNOUNCE_CHANNEL_ID)
    role = guild.get_role(ROLE_ID)
    if not channel or not role:
        return
    # move to inactive category initially
    inactive_cat = await find_channel(guild, INACTIVE_CATEGORY_ID)
    if isinstance(inactive_cat, discord.CategoryChannel):
        await channel.edit(category=inactive_cat)
    # hide channel
    perms = channel.permissions_for(guild.me)
    if perms.manage_channels and guild.me.top_role.position > role.position:
        await channel.set_permissions(guild.default_role, view_channel=False)
        await channel.set_permissions(role, view_channel=False)

async def safe_set_permissions(channel: discord.TextChannel, guild: discord.Guild, view: bool):
    role = guild.get_role(ROLE_ID)
    if not channel.permissions_for(guild.me).manage_channels:
        return
    if view:
        await channel.set_permissions(guild.default_role, overwrite=None)
        await channel.set_permissions(role, view_channel=True)
    else:
        await channel.set_permissions(guild.default_role, view_channel=False)
        await channel.set_permissions(role, view_channel=False)

async def deferred_cleanup(channel, guild):
    await asyncio.sleep(600)
    await finish_cleanup(channel, guild)

@bot.event
async def on_ready():
    global first_ready
    if not first_ready:
        first_ready = True
        print(f'Logged in as {bot.user} (ID: {bot.user.id})')
        await initialize_channel()
        await bot.tree.sync()                # clears old global commands
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
            active_cat = await find_channel(guild, ACTIVE_CATEGORY_ID)
            if isinstance(active_cat, discord.CategoryChannel):
                await channel.edit(category=active_cat)
            await safe_set_permissions(channel, guild, True)
            prefix = '' if TEST_MODE else f'<@&{ROLE_ID}> '
            await channel.send(prefix + f'The chase is on: {link}\nUse `/bingo` to get your card!')
            last_announced_video_id = vid
    else:
        if last_announced_video_id is not None:
            missing_misses += 1
            if missing_misses >= 2:
                last_announced_video_id = None
                missing_misses = 0
                # send a dynamic “in 10 minutes” timestamp
                await channel.send(f"Stream is over — moving channel {rel_ts(600)}")
                bot.loop.create_task(deferred_cleanup(channel, guild))


async def finish_cleanup(channel: discord.TextChannel, guild: discord.Guild):
    inactive_cat = await find_channel(guild, INACTIVE_CATEGORY_ID)
    if isinstance(inactive_cat, discord.CategoryChannel):
        await channel.edit(category=inactive_cat)
    await safe_set_permissions(channel, guild, False)
    await channel.send("Channel moved to inactive category and hidden.")

# slash commands
@bot.tree.command(guild=GUILD, name='bingo', description='Generate a bingo card for the current chase')
async def bingo(interaction: discord.Interaction):
    now = datetime.datetime.utcnow()
    uid = interaction.user.id
    first = last_bingo_usage.get(uid)
    regen = bingo_regenerated.get(uid, False)

    # reset window if expired or first ever
    if first is None or (now - first).total_seconds() >= 3600:
        last_bingo_usage[uid] = now
        bingo_regenerated[uid] = False
        regen = False

    # if inside window but already used regen, block
    if first is not None and (now - first).total_seconds() < 3600 and regen:
        remaining = 3600 - (now - first).total_seconds()
        mins = int(remaining // 60) + 1
        return await interaction.response.send_message(
            f'You’ve already regenerated once. Try again in {mins} minute(s).',
            ephemeral=True
        )

    # mark regen if this is the second draw in the hour
    is_regen = False
    if first is not None and (now - first).total_seconds() < 3600:
        bingo_regenerated[uid] = True
        is_regen = True

    # generate image
    spaces = load_bingo_spaces()
    if len(spaces) < 25:
        return await interaction.response.send_message('Not enough bingo spaces configured.', ephemeral=True)

    image_bytes = generate_bingo_image(spaces, interaction.user.display_name)

    # attempt DM
    try:
        dm_file = discord.File(io.BytesIO(image_bytes), filename='bingo.png')
        await interaction.user.send("Here's your bingo board:", file=dm_file)
        # build response text
        response_text = 'I’ve DMed you your bingo card! 📬'
        if is_regen:
            response_text += "\n\n⚠️ Please only play one board. You cannot regenerate again until the hour is up."
        await interaction.response.send_message(response_text, ephemeral=True)

    except discord.Forbidden:
        # fallback ephemeral + warning if regen
        fallback_file = discord.File(io.BytesIO(image_bytes), filename='bingo.png')
        warn = "\n\n⚠️ Please only play one board. You cannot regenerate again until the hour is up." if is_regen else ""
        await interaction.response.send_message(
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

def load_bingo_spaces(path: str = 'spaces.ini') -> list[str]:
    # Read spaces.ini (no-value entries allowed)
    spaces_cfg = configparser.ConfigParser(allow_no_value=True)
    spaces_cfg.optionxform = str  # preserve case in space text
    spaces_cfg.read(os.path.join(os.path.dirname(__file__), path))

    easy_spaces = list(spaces_cfg['Easy'])
    medium_spaces = list(spaces_cfg['Medium'])
    hard_spaces = list(spaces_cfg['Hard'])

    # Weighted selection: fewer 'Hard', more 'Easy'
    easy_count = 17
    medium_count = 5
    hard_count = 3

    selected = []
    selected += random.sample(easy_spaces, min(easy_count, len(easy_spaces)))
    selected += random.sample(medium_spaces, min(medium_count, len(medium_spaces)))
    selected += random.sample(hard_spaces, min(hard_count, len(hard_spaces)))

    # Fill remaining slots (if any) from all categories
    all_spaces = easy_spaces + medium_spaces + hard_spaces
    remaining = list(set(all_spaces) - set(selected))
    while len(selected) < 25 and remaining:
        choice = random.choice(remaining)
        selected.append(choice)
        remaining.remove(choice)

    random.shuffle(selected)
    return selected

def generate_bingo_image(spaces: list[str], username: str) -> bytes:
    """
    Generate a styled 5x5 bingo board image with a central free space.
    """
    # Layout settings
    margin = 20
    header_height = 60
    footer_height = 40
    board_size = 600  # base width for canvas calculations

    # Canvas dimensions
    width = board_size + 2 * margin
    # compute cell size to fill width between margins
    cell = (width - 2 * margin) // 5
    board_draw_size = cell * 5
    total_height = header_height + board_draw_size + footer_height
    img = Image.new('RGBA', (width, total_height), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Load fonts
    try:
        title_font = ImageFont.truetype('arial.ttf', size=24)
        cell_font = ImageFont.truetype('arial.ttf', size=16)
        footer_font = ImageFont.truetype('arial.ttf', size=18)
    except IOError:
        title_font = ImageFont.load_default()
        cell_font = ImageFont.load_default()
        footer_font = ImageFont.load_default()

    # Watermark
    watermark_text = 'B I N G O'
    bbox = draw.textbbox((0, 0), watermark_text, font=title_font)
    wmw, wmh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    watermark = Image.new('RGBA', (width, total_height), (255, 255, 255, 0))
    wm_draw = ImageDraw.Draw(watermark)
    x_w = (width - wmw) // 2
    y_w = (total_height - wmh) // 2
    wm_draw.text((x_w, y_w), watermark_text, font=title_font, fill=(0, 0, 0, 40))
    wm_rot = watermark.rotate(-45, expand=1)
    # center rotated watermark
    x_off = (width - wm_rot.width) // 2
    y_off = (total_height - wm_rot.height) // 2
    img.paste(wm_rot, (x_off, y_off), wm_rot)

    # Header with date
    today = datetime.date.today().strftime('%B %d, %Y')
    header_text = f"LA Chase Bingo - {today}"
    bbox = draw.textbbox((0, 0), header_text, font=title_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) / 2, (header_height - th) / 2), header_text, font=title_font, fill='black')

    # Prepare board data with center free space
    free_space_text = 'Takes place in Los Angeles FREE SPACE'
    # Ensure free space isn't duplicated
    available = [s for s in spaces if s != free_space_text]
    # Pick 24 random spaces
    chosen = random.sample(available, 24)
    # Insert free space at center index 12 (0-based)
    chosen.insert(12, free_space_text)

    # Draw cells
    start_y = header_height
    for idx, text in enumerate(chosen):
        row, col = divmod(idx, 5)
        x0 = margin + col * cell
        y0 = start_y + row * cell
        x1, y1 = x0 + cell, y0 + cell
        draw.rectangle([x0, y0, x1, y1], fill=(245, 245, 245, 255), outline='gray', width=1)
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

    # Save as PNG bytes
    final = img.convert('RGB')
    buf = io.BytesIO()
    final.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


if __name__ == '__main__':
    bot.run(discord_token)

