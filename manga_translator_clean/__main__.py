from web.app import app

if __name__ == "__main__":
    # mirrors FLASK_DEBUG=1 python web/app.py behavior
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)