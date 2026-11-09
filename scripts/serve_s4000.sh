#!/bin/bash
# Launch the product server from the environment file used by systemd.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE=${S4000_ENV_FILE:-/etc/s4000-moe/serve.env}
VENV=${S4000_VENV:-/opt/s4000-moe/venv}
PREFIX=${S4000_PREFIX:-/opt/s4000-moe}

if [ -f "$ENV_FILE" ] && [ -r "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi
# Under systemd the env file is read by PID 1 as root and passed as
# Environment=; the service user may not be able to read the 600 file itself.

: "${NANOVLLM_MODEL:?NANOVLLM_MODEL is not set; edit $ENV_FILE}"
if [ -n "${S4000_BIN:-}" ] && [ -x "$S4000_BIN" ]; then
  exec "$S4000_BIN"
elif [ -x "$VENV/bin/s4000-server" ]; then
  exec "$VENV/bin/s4000-server"
elif [ -x "$PREFIX/bin/s4000-server" ]; then
  exec "$PREFIX/bin/s4000-server"
fi
echo "s4000-server not found; run scripts/install_s4000.sh first" >&2
exit 1