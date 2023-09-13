from celery import Celery
from multiprocessing import shared_memory
import numpy as np
from PIL import Image
import io
import base64
from redis import Redis
import json
import time
from inference.models.utils import get_roboflow_model
import pyvips

r = Redis(host="inference-redis", port="6379", decode_responses=True)
app = Celery("tasks", broker="redis://inference-redis:6379")
app.conf.result_backend = "redis://inference-redis:6379/0"

model = get_roboflow_model("melee/5", "Nw3QZal3hhwHP5npbWmw")


def decode_base64(base64_string):
    decoded_string = io.BytesIO(base64.b64decode(base64_string))
    img = Image.open(decoded_string)
    return img


@app.task(queue="pre")
def preprocess(model_name, image, id_, request_time):
    image = decode_base64(image)
    image.draft("RGB", (640, 640))
    image.load()
    image, img_dims = model.preprocess(image)
    image = image[0]
    shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
    shared = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
    shared[:] = image[:]
    shm.close()
    return_vals = {
        "chunk_name": shm.name,
        "image_shape": image.shape,
        "image_dtype": image.dtype.name,
        "id": id_,
        "image_dim": img_dims[0],
    }
    return_vals = json.dumps(return_vals)
    r.zadd(f"infer:{model_name}", {return_vals: request_time})
    r.hincrby(f"requests", model_name, 1)


@app.task(queue="post")
def postprocess(args, id_, dim):
    shm = shared_memory.SharedMemory(name=args["chunk_name"])
    image = np.ndarray(args["image_shape"], dtype=args["image_dtype"], buffer=shm.buf)
    results = model.postprocess(np.array([image]), np.array([dim]))
    results = model.make_response(results, np.array([dim]))[0]
    results = results.json(exclude_none=True, by_alias=True)
    r.set(f"results:{id_}", json.dumps(results))
    r.set(f"status:{id_}", 1)
    shm.close()
    shm.unlink()
