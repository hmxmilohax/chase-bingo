#!/bin/sh
set -eu

if [ ! -f /config/config.ini ]; then
  echo "[chase-bingo] ERROR: /config/config.ini not found (bind-mount it)."
  exit 1
fi

cp /config/config.ini /app/config.ini

cd /app
exec python chase-bingo.py
