#!/bin/sh
set -eu

: "${REPO_URL:=https://github.com/hmxmilohax/chase-bingo}"
: "${APP_DIR:=/opt/chase-bingo}"
: "${BRANCH:=main}"

mkdir -p "$APP_DIR"
mkdir -p /config

git config --global --add safe.directory "$APP_DIR" || true

if [ ! -d "$APP_DIR/.git" ]; then
  echo "[chase-bingo] Cloning repo..."
  find "$APP_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
else
  echo "[chase-bingo] Updating repo..."
  git -C "$APP_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$APP_DIR" reset --hard "origin/$BRANCH"
fi

if [ ! -f /config/config.ini ]; then
  echo "[chase-bingo] ERROR: /config/config.ini not found (bind-mount it)."
  exit 1
fi

cp /config/config.ini "$APP_DIR/config.ini"

cd "$APP_DIR"
exec python chase-bingo.py
