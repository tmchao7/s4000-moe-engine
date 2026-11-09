#!/bin/bash
# Install s4000-moe-engine into a venv without touching the vendor MUSA torch.
#
# The MUSA driver + torch_musa wheel must already be installed in the host
# Python. This script never installs torch: pip would otherwise replace the
# vendor build with a CUDA wheel.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=${S4000_PREFIX:-/opt/s4000-moe}
VENV=${S4000_VENV:-$PREFIX/venv}
USER_NAME=${S4000_USER:-s4000}
PYTHON=${PYTHON:-python3}
INSTALL_SYSTEMD=auto

usage() {
  cat <<EOF
Usage: bash scripts/install_s4000.sh [--prefix DIR] [--python PYTHON] [--no-systemd]

  --prefix DIR    install prefix (default: $PREFIX)
  --python PATH   host Python with torch_musa (default: python3)
  --no-systemd    do not install/enable the systemd unit
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX=$2; VENV=${S4000_VENV:-$PREFIX/venv}; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --no-systemd) INSTALL_SYSTEMD=no; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

echo "[install] app=$ROOT prefix=$PREFIX venv=$VENV"

"$PYTHON" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"Python 3.10-3.12 required, got {sys.version.split()[0]}")
import torch, torch_musa
if not torch.musa.is_available():
    raise SystemExit("torch.musa is not available; install the vendor torch_musa wheel first")
print(f"[install] host python {sys.version.split()[0]} torch {torch.__version__} MUSA OK")
PY

mkdir -p "$PREFIX" /etc/s4000-moe
VENV_OK=0
if [ -x "$VENV/bin/python" ] && [ -x "$VENV/bin/pip" ]; then
  VENV_OK=1
elif "$PYTHON" -m venv --system-site-packages "$VENV" 2>/dev/null; then
  VENV_OK=1
else
  echo "[install] python venv unavailable; falling back to an isolated --target install"
  TARGET="$PREFIX/lib"
  mkdir -p "$TARGET" "$PREFIX/bin"
  # --target does not see system packages, so install the project and the
  # small serve deps with --no-deps and leave torch/numpy to the vendor/system
  # stack (installing numpy here shadows the wheel torch_musa was built
  # against).  Transformers/tokenizers are version-coupled: installing the
  # latest transformers with --no-deps pulled tokenizers>=0.23 while the host
  # had 0.22.x and broke import.  Pin both to the host versions instead.
  HOST_TRANSFORMERS=$("$PYTHON" -c 'import transformers; print(transformers.__version__)')
  HOST_TOKENIZERS=$("$PYTHON" -c 'import tokenizers; print(tokenizers.__version__)')
  "$PYTHON" -m pip install --no-deps --upgrade --target "$TARGET" "$ROOT"
  for pkg in "fastapi>=0.115" "uvicorn>=0.30" "prometheus-client>=0.21" \
             "transformers==${HOST_TRANSFORMERS}" "tokenizers==${HOST_TOKENIZERS}" \
             "tqdm" "xxhash"; do
    "$PYTHON" -m pip install --no-deps --upgrade --target "$TARGET" "$pkg"
  done
  cat > "$PREFIX/bin/s4000-server" <<EOF
#!/bin/bash
export PYTHONPATH="$TARGET:\${PYTHONPATH:-}"
exec "$PYTHON" -m nanovllm.entrypoints.openai.serve "\$@"
EOF
  chmod 755 "$PREFIX/bin/s4000-server"
fi
if [ "$VENV_OK" -eq 1 ]; then
  "$VENV/bin/pip" install --no-deps -e "$ROOT"
  "$VENV/bin/pip" install -r "$ROOT/requirements-serve.txt"
  "$VENV/bin/python" - <<'PY'
import nanovllm, torch
assert torch.musa.is_available(), "torch.musa unavailable in the service venv"
print(f"[install] nanovllm {nanovllm.__version__} import OK")
PY
  S4000_BIN="$VENV/bin/s4000-server"
else
  PYTHONPATH="$TARGET" "$PYTHON" - <<'PY'
import fastapi, uvicorn, prometheus_client, tqdm, xxhash
import transformers, torch
import nanovllm
assert torch.musa.is_available(), "torch.musa unavailable in the target install"
print(f"[install] nanovllm {nanovllm.__version__} import OK (target)")
PY
  S4000_BIN="$PREFIX/bin/s4000-server"
fi

if [ ! -f /etc/s4000-moe/serve.env ]; then
  install -m 600 "$ROOT/deploy/serve.env.example" /etc/s4000-moe/serve.env
  echo "[install] wrote /etc/s4000-moe/serve.env (edit NANOVLLM_MODEL before start)"
else
  echo "[install] keeping existing /etc/s4000-moe/serve.env"
fi

if [ "$INSTALL_SYSTEMD" != "no" ]; then
  if [ "$(id -u)" -eq 0 ] && command -v systemctl >/dev/null 2>&1; then
    if ! getent group "$USER_NAME" >/dev/null 2>&1; then
      groupadd --system "$USER_NAME"
    fi
    if ! id -u "$USER_NAME" >/dev/null 2>&1; then
      useradd --system --gid "$USER_NAME" --home-dir "$PREFIX" \
        --shell /usr/sbin/nologin "$USER_NAME"
    fi
    if [[ "$ROOT" == "$PREFIX"* ]]; then
      chown -R "$USER_NAME:$USER_NAME" "$PREFIX"
    else
      echo "[install] WARN: app is not under $PREFIX; ensure $USER_NAME can read $ROOT"
    fi
    sed -e "s|@USER@|$USER_NAME|g" \
        -e "s|@APP_DIR@|$ROOT|g" \
        -e "s|@VENV@|$VENV|g" \
        -e "s|@BIN@|$S4000_BIN|g" \
        "$ROOT/deploy/s4000-moe.service" > /etc/systemd/system/s4000-moe.service
    systemctl daemon-reload
    echo "[install] systemd unit installed"
  else
    echo "[install] skip systemd (not root or no systemctl); run as root to install the unit"
  fi
fi

echo
echo "[install] next:"
echo "  1. sudo editor /etc/s4000-moe/serve.env"
echo "  2. sudo systemctl enable --now s4000-moe"
echo "  3. bash $ROOT/scripts/smoke_serve.sh --base-url http://127.0.0.1:8000"