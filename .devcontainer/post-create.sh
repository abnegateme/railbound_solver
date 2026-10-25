#!/usr/bin/env bash
set -euo pipefail

CODEX_DIR="/home/vscode/.codex"
CONFIG="$CODEX_DIR/config.toml"

sudo chown -R "$(id -u):$(id -g)" /workspace "$CODEX_DIR"
sudo chmod -R u+rwX /workspace "$CODEX_DIR"

mkdir -p "$CODEX_DIR"
touch "$CONFIG"

TMP="$(mktemp "$CODEX_DIR/config.toml.XXXXXX")"

cat > "$TMP" <<'EOF'
sandbox_mode = "danger-full-access"
approval_policy = "on-request"

EOF

grep -Ev \
  '^[[:space:]]*(sandbox_mode|approval_policy)[[:space:]]*=' \
  "$CONFIG" >> "$TMP" || true

mv "$TMP" "$CONFIG"