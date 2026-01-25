FROM python:3.12-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U \
    discord.py aiohttp beautifulsoup4 pillow

COPY . .
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
