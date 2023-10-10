from inference.core.managers.base import Model, ModelManager
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.env import REDIS_HOST, REDIS_PORT
from redis import Redis

LOCK_NAME = "locks:model_load:{}"
class LockedLoadModelManagerDecorator(ModelManagerDecorator):
    def __init__(self, model_manager: ModelManager, redis: Redis=None):
        super().__init__(model_manager)
        if redis is None:
            redis = Redis(host=REDIS_HOST, port=REDIS_PORT)
        self.redis = redis

    def add_model(self, model_id: str, model: Model):
        return super().add_model(model_id, model)