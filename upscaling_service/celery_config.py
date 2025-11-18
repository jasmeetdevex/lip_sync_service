import os
from celery import Celery

celery = Celery(__name__)

celery.conf.update(
    broker_url=os.getenv("CELERY_BROKER_URL", "memory://"),
    result_backend=os.getenv("CELERY_RESULT_BACKEND", "cache+memory://"),
    accept_content=['json'],
    task_serializer='json',
    result_serializer='json',
    timezone='UTC',
    task_track_started=True,
    task_time_limit=30 * 60,
)

import app