from flask import Flask
from extensions import mongo
import logging
import os
logging.basicConfig(level=logging.INFO)

# Import Celery app
from celery_config import celery

def create_app():
    """Application factory"""
    app = Flask(__name__)

    # Configuration
    ENV = os.getenv("ENV", "development")  # default to development
    MONGO_LIVE_URI = os.getenv("MONGO_LIVE_URI")
    MONGO_LOCAL_URI = "mongodb+srv://narratixdev:sltgyS9EjJPqfXQw@narratixcluster.00fijbr.mongodb.net/narratixdb?retryWrites=true&w=majority&appName=NarratixCluster"

    if ENV == "production":
        app.config["MONGO_URI"] = MONGO_LIVE_URI
    else:
        app.config["MONGO_URI"] = MONGO_LOCAL_URI

    # Initialize MongoDB
    mongo.init_app(app)

    # Update Celery with app context
    celery.conf.update(app=app)

    # ⭐ CRITICAL: Add Flask app context to Celery tasks
    class ContextTask(celery.Task):
        """Make celery tasks work with Flask app context"""
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    app.celery = celery

    # Register blueprints
    from routes import lip_sync_routes
    app.register_blueprint(lip_sync_routes.bp)

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5001)
