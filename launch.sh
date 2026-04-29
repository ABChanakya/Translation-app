#!/bin/bash
# Quick launcher for Auto Manga Translation

cd ~/chanakya/Translation_tool-2/manga_translator_clean

# Use the venv Python, fall back to python3, then python
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║            🎌 AUTO MANGA TRANSLATION - LAUNCHER                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "What would you like to do?"
echo ""
echo "1) 🌐 Start Web Interface only"
echo "2) 🌐 Start Web + LaMa Inpainting  ← recommended for best quality"
echo "3) 🎨 Start LaMa Inpainting Service only (port 5001)"
echo "4) ⚡ Start FastAPI async service (port 8000)"
echo "5) 🏋️  Train a Model (Admin/developer mode)"
echo "6) 📊 View System Status"
echo "7) 📚 Open Documentation"
echo "8) 🛠️  Run Data Utilities"
echo "9) 📖 Show Quickstart"
echo "0) 🚪 Exit"
echo ""
read -p "Enter your choice (0-9): " choice

is_port_in_use() {
    local port="$1"
    lsof -i TCP:"$port" >/dev/null 2>&1
}

case $choice in
    1)
        echo ""
        echo "🌐 Starting Web Interface..."
        echo "   Visit: http://localhost:5000"
        echo ""
        $PYTHON web/app.py
        ;;
    2)
        echo ""
        if is_port_in_use 5001; then
            echo "🎨 LaMa inpainting service is already running on port 5001."
            echo "   Reusing the existing service."
        else
            echo "🎨 Starting LaMa inpainting service in the background..."
            bash lama_service/start_service.sh &
            LAMA_PID=$!
            echo "   LaMa PID: $LAMA_PID  (listening on port 5001)"
            echo "   Waiting 10 seconds for the model to load..."
            sleep 10
        fi
        echo ""
        echo "🌐 Starting Web Interface..."
        echo "   Visit: http://localhost:5000"
        echo "   Press Ctrl+C to stop both services."
        echo ""
        if [ -n "$LAMA_PID" ]; then
            trap "echo ''; echo 'Stopping LaMa service...'; kill $LAMA_PID 2>/dev/null; exit" INT TERM
        fi
        $PYTHON web/app.py
        if [ -n "$LAMA_PID" ]; then
            kill $LAMA_PID 2>/dev/null
        fi
        ;;
    3)
        echo ""
        if is_port_in_use 5001; then
            echo "🎨 LaMa inpainting service is already running on port 5001."
            echo "   Reuse the existing service or stop it before starting a new one."
            exit 0
        fi
        echo "🎨 Starting LaMa inpainting service on http://0.0.0.0:5001..."
        bash lama_service/start_service.sh
        ;;
    4)
        echo ""
        if is_port_in_use 5001; then
            echo "🎨 LaMa inpainting service is already running on port 5001."
            echo "   Reusing the existing service."
        else
            echo "🎨 Starting LaMa inpainting service in the background..."
            bash lama_service/start_service.sh &
            LAMA_PID=$!
            echo "   LaMa PID: $LAMA_PID  (listening on port 5001)"
            echo "   Waiting 10 seconds for the model to load..."
            sleep 10
        fi
        echo ""
        echo "⚡ Starting FastAPI async service on http://0.0.0.0:8000..."
        echo "   Web UI:    http://localhost:8000"
        echo "   API Docs:  http://localhost:8000/docs"
        echo "   Endpoints: POST /translate  POST /translate/vlm"
        echo "   Press Ctrl+C to stop both services."
        echo ""
        if [ -n "$LAMA_PID" ]; then
            trap "echo ''; echo 'Stopping LaMa service...'; kill $LAMA_PID 2>/dev/null; exit" INT TERM
        fi
        $PYTHON -m uvicorn src.fastapi_service:app --host 0.0.0.0 --port 8000
        if [ -n "$LAMA_PID" ]; then
            kill $LAMA_PID 2>/dev/null
        fi
        ;;
    5)
        echo ""
        echo "🏋️  Available training scripts:"
        echo "   1. advanced_train_yolo.py"
        echo "   2. train_and_eval.py"
        echo "   3. create_pseudo_labels.py"
        echo ""
        read -p "Choose (1-3): " train_choice
        case $train_choice in
            1) $PYTHON training/advanced_train_yolo.py ;;
            2) $PYTHON training/train_and_eval.py ;;
            3) $PYTHON training/create_pseudo_labels.py --help ;;
            *) echo "Invalid choice" ;;
        esac
        ;;
    6)
        echo ""
        $PYTHON verify_all.py | head -50
        echo ""
        echo "📖 Read full report: $PYTHON verify_all.py"
        ;;
    7)
        echo ""
        echo "📚 Available Documentation:"
        printf '%s\n' \
          "README.md" \
          "QUICKSTART.md" \
          "DEVELOPER_GUIDE.md" \
          "FUTURE_ACCESSIBILITY_AND_SCALING.md" \
          "data/README.md" \
          "colorization/readme.md"
        echo ""
        read -p "Enter filename to open (e.g., QUICKSTART.md): " doc_file
        if [ -f "$doc_file" ]; then
            less "$doc_file"
        else
            echo "File not found!"
        fi
        ;;
    8)
        echo ""
        echo "🛠️  Data Utilities:"
        echo "   1. dedupe.py - Remove duplicates"
        echo "   2. counts.py - Dataset statistics"
        echo "   3. rawkuma_scraper.py - Download manga"
        echo "   4. rename_seq.py - Rename files"
        echo ""
        read -p "Choose (1-4): " util_choice
        case $util_choice in
            1) $PYTHON data/utils/dedupe.py --help ;;
            2) $PYTHON data/utils/counts.py --help ;;
            3) $PYTHON data/scrapers/rawkuma_scraper.py --help ;;
            4) $PYTHON data/utils/rename_seq.py --help ;;
            *) echo "Invalid choice" ;;
        esac
        ;;
    9)
        echo ""
        cat QUICKSTART.md | less
        ;;
    0)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice!"
        ;;
esac
