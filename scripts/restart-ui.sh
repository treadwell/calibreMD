#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[ui] rebuilding and restarting calibre recommendation UI..."
docker compose -f docker-compose.ui.yml up -d --build

echo "[ui] service status:"
docker compose -f docker-compose.ui.yml ps

echo "[ui] logs (tail 30):"
docker compose -f docker-compose.ui.yml logs --tail=30 calibre-reco-ui

echo "[ui] open: http://localhost:8780"
