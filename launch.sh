#!/bin/bash
# Quick launcher for Auto Manga Translation

cd ~/chanakya/Translation_tool-2/manga_translator_clean

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║            🎌 AUTO MANGA TRANSLATION - LAUNCHER                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "What would you like to do?"
echo ""
echo "1) 🌐 Start Web Interface only"
echo "2) 🌐 Start Web + LaMa Inpainting  ← recommended for best quality"
echo "3) 🎨 Start LaMa Inpainting Service only (port 5001)"
echo "4) 🏋️  Train a Model (Admin/developer mode)"
echo "5) 📊 View System Status"
echo "6) 📚 Open Documentation"
echo "7) 🛠️  Run Data Utilities"
echo "8) 📖 Show Quickstart"
echo "9) 🚪 Exit"
echo ""
read -p "Enter your choice (1-9): " choice

case $choice in
    1)
        echo ""
        echo "🌐 Starting Web Interface..."
        echo "   Visit: http://localhost:5000"
        echo ""
        python web/app.py
        ;;
    2)
        echo ""
        echo "🎨 Starting LaMa inpainting service in the background..."
        bash lama_service/start_service.sh &
        LAMA_PID=$!
        echo "   LaMa PID: $LAMA_PID  (listening on port 5001)"
        echo "   Waiting 10 seconds for the model to load..."
        sleep 10
        echo ""
        echo "🌐 Starting Web Interface..."
        echo "   Visit: http://localhost:5000"
        echo "   Press Ctrl+C to stop both services."
        echo ""
        trap "echo ''; echo 'Stopping LaMa service...'; kill $LAMA_PID 2>/dev/null; exit" INT TERM
        python web/app.py
        kill $LAMA_PID 2>/dev/null
        ;;
    3)
        echo ""
        echo "🎨 Starting LaMa inpainting service on http://0.0.0.0:5001..."
        bash lama_service/start_service.sh
        ;;
    4)
        echo ""
        echo "🏋️  Available training scripts:"
        echo "   1. advanced_train_yolo.py"
        echo "   2. train_and_eval.py"
        echo "   3. create_pseudo_labels.py"
        echo ""
        read -p "Choose (1-3): " train_choice
        case $train_choice in
            1) python training/advanced_train_yolo.py ;;
            2) python training/train_and_eval.py ;;
            3) python training/create_pseudo_labels.py --help ;;
            *) echo "Invalid choice" ;;
        esac
        ;;
    5)
        echo ""
        python verify_all.py | head -50
        echo ""
        echo "📖 Read full report: python verify_all.py"
        ;;
    6)
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
    7)
        echo ""
        echo "🛠️  Data Utilities:"
        echo "   1. dedupe.py - Remove duplicates"
        echo "   2. counts.py - Dataset statistics"
        echo "   3. rawkuma_scraper.py - Download manga"
        echo "   4. rename_seq.py - Rename files"
        echo ""
        read -p "Choose (1-4): " util_choice
        case $util_choice in
            1) python data/utils/dedupe.py --help ;;
            2) python data/utils/counts.py --help ;;
            3) python data/scrapers/rawkuma_scraper.py --help ;;
            4) python data/utils/rename_seq.py --help ;;
            *) echo "Invalid choice" ;;
        esac
        ;;
    8)
        echo ""
        cat QUICKSTART.md | less
        ;;
    9)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice!"
        ;;
esac
