from multiprocessing import shared_memory
import logging
import numpy as np
from PIL import Image
import io
import base64
from redis import Redis
import json
import time
from tasks import postprocess
from inference.models.utils import get_roboflow_model

r = Redis(host="inference-redis", port="6379", decode_responses=True)
BATCH_SIZE = 64
logging.basicConfig(level=logging.INFO)

class InferServer:
    def __init__(self):
        self.model = get_roboflow_model("melee/5", "Nw3QZal3hhwHP5npbWmw")

    def get_batch(self, model_names):
        batches = [r.zrange(f"infer:{m}", 0, BATCH_SIZE - 1, withscores=True) for m in model_names]
        now = time.time()
        average_ages = [np.mean([float(b[1]) - now for b in batch]) for batch in batches]
        lengths = [len(batch) / BATCH_SIZE for batch in batches]
        fitnesses = [age / 30 + length for age, length in zip(average_ages, lengths)]
        model_index = fitnesses.index(max(fitnesses))
        batch = batches[model_index]
        selected_model = model_names[model_index]
        r.zrem(f"infer:{selected_model}", *[b[0] for b in batch])
        r.hincrby(f"requests", selected_model, -len(batch))
        batch = [json.loads(b[0]) for b in batch]
        return batch

    def infer_loop(self):
        while True:
            request_counts = r.hgetall("requests")
            model_names = [model_name for model_name, count in request_counts.items() if int(count) > 0]
            if not model_names:
                time.sleep(0.1)
                continue
            batch = self.get_batch(model_names)
            logging.info(f"BATCH SIZE {len(batch)}")
            images = []
            dims = []
            shms = []
            for b in batch:
                shm = shared_memory.SharedMemory(name=b["chunk_name"])
                image = np.ndarray(b["image_shape"], dtype=b["image_dtype"], buffer=shm.buf)
                images.append(image)
                dims.append(b["image_dim"])
                shms.append(shm)
            outputs = self.model.predict(images)
            del images
            for shm in shms:
                shm.close()
                shm.unlink()
            for output, b, dim in zip(outputs, batch, dims):
                info = self.write_response(output)
                postprocess.s(info, b["id"], dim).delay()
            
    def load_image(self, args):
        shm = shared_memory.SharedMemory(name=args["chunk_name"])
        return image

    def write_response(self, im_arr):
        shm2 = shared_memory.SharedMemory(create=True, size=im_arr.nbytes)
        shared = np.ndarray(im_arr.shape, dtype=im_arr.dtype, buffer=shm2.buf)
        shared[:] = im_arr[:]
        return_val = {"chunk_name": shm2.name, "image_shape": im_arr.shape, "image_dtype": im_arr.dtype.name}
        shm2.close()
        return return_val


if __name__ == "__main__":
    InferServer().infer_loop()