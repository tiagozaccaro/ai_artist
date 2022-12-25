from celery import Celery

# Broker settings.
broker_backend = "sqla+sqlite://aiartist.db"
cache_backend = "sqla+sqlite://aiartist.db"
result_backend = "sqla+sqlite://aiartist.db"

celery = Celery(__name__, broker=broker_backend, backend=result_backend)
celery.conf.update(celery_accept_content=["json"], result_serializer="json")
celery.conf.update(celery_task_serializer="json", task_track_started=True)
