#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Auto Manga Translation — Cloud Instance Bootstrap
# Run once on a fresh GPU cloud instance (RunPod / Vast.ai / Lambda / EC2).
# Installs everything needed, then starts Ollama + LaMa + the Flask app.
#
#   bash cloud_setup.sh
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$PROJECT_ROOT/manga_translator_clean"
VENV="$APP_ROOT/.venv"
LAMA_DIR="$APP_ROOT/lama_service"
LAMA_VENV="$LAMA_DIR/venv_lama"
LOG_DIR="$APP_ROOT/logs"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()      { echo -e "${GREEN}✅  $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠️   $*${NC}"; }
err()     { echo -e "${RED}❌  $*${NC}"; exit 1; }
section() { echo -e "\n${CYAN}━━━  $*  ━━━${NC}"; }

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     🎌  AUTO MANGA TRANSLATION — CLOUD BOOTSTRAP               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
section "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    python3 python3-venv python3-pip \
    python3-dev build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    curl wget git fonts-dejavu-core \
    > /dev/null
ok "System packages installed"

# ── 2. Ollama ─────────────────────────────────────────────────────────────────
section "Ollama"
if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
else
    ok "Ollama already installed ($(ollama --version 2>/dev/null || echo 'unknown version'))"
fi

# Start Ollama server in the background
if ! pgrep -x ollama > /dev/null 2>&1; then
    nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    echo -n "   Waiting for Ollama to start"
    for i in $(seq 1 15); do
        sleep 1; echo -n "."
        curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1 && break
    done
    echo ""
fi
ok "Ollama running"

# Pull Gemma3 12B (this takes a while on first run)
if ollama list 2>/dev/null | grep -q "gemma3:12b"; then
    ok "Gemma3 12B already pulled"
else
    echo "Pulling Gemma3 12B (this may take several minutes)..."
    ollama pull gemma3:12b
    ok "Gemma3 12B ready"
fi

# ── 3. Main app Python venv ───────────────────────────────────────────────────
section "Main app Python environment"
if [ ! -f "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install -q --upgrade pip
pip install -q -r "$APP_ROOT/requirements.txt"
ok "Main app dependencies installed"

# ── 4. LaMa inpainting service ────────────────────────────────────────────────
section "LaMa inpainting service"
if [ ! -f "$LAMA_VENV/bin/python" ]; then
    python3 -m venv "$LAMA_VENV"
fi

"$LAMA_VENV/bin/pip" install -q --upgrade pip
"$LAMA_VENV/bin/pip" install -q -r "$LAMA_DIR/requirements.txt"
touch "$LAMA_VENV/deps_installed.marker"
ok "LaMa dependencies installed"

# Start LaMa service
if curl -sf http://127.0.0.1:5001/health > /dev/null 2>&1; then
    ok "LaMa service already running"
else
    (
        source "$LAMA_VENV/bin/activate"
        cd "$LAMA_DIR"
        export LAMA_DEVICE="${LAMA_DEVICE:-auto}"
        nohup gunicorn --workers 1 --bind 0.0.0.0:5001 --timeout 300 app:app \
            > "$LOG_DIR/lama.log" 2>&1 &
    )
    echo -n "   Waiting for LaMa model to load"
    for i in $(seq 1 30); do
        sleep 2; echo -n "."
        curl -sf http://127.0.0.1:5001/health > /dev/null 2>&1 && break
    done
    echo ""
    if curl -sf http://127.0.0.1:5001/health > /dev/null 2>&1; then
        ok "LaMa service ready on port 5001"
    else
        warn "LaMa still loading — see $LOG_DIR/lama.log"
    fi
fi

# ── 5. GPU check ──────────────────────────────────────────────────────────────
section "GPU"
python - <<'EOF'
import torch, sys
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f"\033[0;32m✅  GPU: {name}  ({mem} GiB VRAM)\033[0m")
else:
    print("\033[1;33m⚠️   No CUDA GPU — running on CPU (translation will be slow)\033[0m")
EOF

# ── 6. Start the Flask app ────────────────────────────────────────────────────
section "Starting web interface"
source "$VENV/bin/activate"
cd "$APP_ROOT"

# Expose on 0.0.0.0:5000 so cloud port-forwarding works
echo ""
echo "  🌐 Web interface → http://0.0.0.0:5000"
echo "  📋 LaMa logs     → $LOG_DIR/lama.log"
echo "  📋 Ollama logs   → $LOG_DIR/ollama.log"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

python web/app.py
