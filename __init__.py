from flask import Flask

from .routes import register_routes


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object("app.config.Config")
    register_routes(app)
    return app
