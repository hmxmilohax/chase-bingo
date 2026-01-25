FROM python:3.12-slim

WORKDIR /opt

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates tzdata \
    fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

# Python deps used by chase-bingo
RUN pip install --no-cache-dir -U \
    discord.py aiohttp beautifulsoup4 pillow

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV REPO_URL="https://github.com/hmxmilohax/chase-bingo"
ENV APP_DIR="/opt/chase-bingo"
ENV BRANCH="main"

ENTRYPOINT ["/entrypoint.sh"]
