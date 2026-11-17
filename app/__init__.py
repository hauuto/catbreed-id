from flask import Flask,  render_template


def create_app():
    app = Flask(__name__, static_folder="static", static_url_path="")

    @app.get("/")
    def index():
        return render_template("index.html")
    @app.get("/health")
    def health():
        return {"status": "ok"}
    return app