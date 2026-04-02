#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Auto Manga Translation — Session Setup Script
# Run once when a new session starts (before launching Claude Code or the app).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$PROJECT_ROOT/manga_translator_clean"
VENV="$APP_ROOT/.venv"
LAMA_DIR="$APP_ROOT/lama_service"
LAMA_VENV="$LAMA_DIR/venv_lama"
LOG_DIR="$APP_ROOT/logs"

mkdir -p "$LOG_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
err()  { echo -e "${RED}❌ $*${NC}"; }

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        🎌 AUTO MANGA TRANSLATION — SESSION SETUP            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Python venv ────────────────────────────────────────────────────────────
if [ -f "$VENV/bin/python" ]; then
    ok "Python venv ready  ($VENV/bin/python)"
else
    warn "venv not found — creating it..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -r "$APP_ROOT/requirements.txt" && ok "Dependencies installed"
fi

# Activate for the rest of this script
source "$VENV/bin/activate"

# ── 2. GPU check ──────────────────────────────────────────────────────────────
if python - <<'EOF' 2>/dev/null
import torch, sys
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f"GPU: {name}  ({mem} GiB VRAM)")
    sys.exit(0)
sys.exit(1)
EOF
then
    ok "$(python - <<'EOF' 2>/dev/null
import torch
name = torch.cuda.get_device_name(0)
mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
print(f"GPU detected: {name}  ({mem} GiB VRAM)")
EOF
)"
else
    warn "No CUDA GPU detected — will run on CPU (slower)"
fi

# ── 3. Ollama / Gemma3 ────────────────────────────────────────────────────────
if pgrep -x ollama > /dev/null 2>&1; then
    ok "Ollama already running"
else
    warn "Ollama not running — starting it..."
    nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    sleep 3
    if pgrep -x ollama > /dev/null 2>&1; then
        ok "Ollama started  (log: $LOG_DIR/ollama.log)"
    else
        err "Ollama failed to start — check $LOG_DIR/ollama.log"
    fi
fi

# Check gemma3:12b model is pulled
if ollama list 2>/dev/null | grep -q "gemma3:12b"; then
    ok "Gemma3 12B model ready"
else
    warn "Gemma3 12B not found — pulling (this may take a while)..."
    ollama pull gemma3:12b && ok "Gemma3 12B pulled" || err "Failed to pull gemma3:12b"
fi

# ── 4. LaMa inpainting service ────────────────────────────────────────────────
if curl -sf http://127.0.0.1:5001/health > /dev/null 2>&1; then
    ok "LaMa service already running on port 5001"
else
    if [ -f "$LAMA_VENV/bin/python" ] && "$LAMA_VENV/bin/python" -c "import simple_lama_inpainting" 2>/dev/null; then
        warn "LaMa service not running — starting it..."
        (
            source "$LAMA_VENV/bin/activate"
            cd "$LAMA_DIR"
            export LAMA_DEVICE="${LAMA_DEVICE:-auto}"
            nohup gunicorn --workers 1 --bind 0.0.0.0:5001 --timeout 300 app:app \
                > "$LOG_DIR/lama.log" 2>&1 &
        )
        echo -n "   Waiting for LaMa to load model"
        for i in $(seq 1 20); do
            sleep 2
            echo -n "."
            if curl -sf http://127.0.0.1:5001/health > /dev/null 2>&1; then
                echo ""
                ok "LaMa service ready on port 5001  (log: $LOG_DIR/lama.log)"
                break
            fi
        done
        if ! curl -sf http://127.0.0.1:5001/health > /dev/null 2>&1; then
            echo ""
            warn "LaMa still loading — it may take a few more seconds. Check $LOG_DIR/lama.log"
        fi
    else
        warn "LaMa venv not set up — run:  bash $LAMA_DIR/start_service.sh  to install & start"
    fi
fi

# ── 5. Port check (nothing already on 5000) ───────────────────────────────────
if ss -ltn 2>/dev/null | grep -q ':5000 '; then
    warn "Port 5000 is already in use — the Flask app may already be running"
else
    ok "Port 5000 is free"
fi

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  Ready. To start the web interface:"
echo "    cd $APP_ROOT"
echo "    source .venv/bin/activate"
echo "    python web/app.py"
echo ""
echo "  Or use the launcher:"
echo "    bash $PROJECT_ROOT/launch.sh  (choose option 1)"
echo "──────────────────────────────────────────────────────────────"
echo ""
