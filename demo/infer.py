from multiprocessing import shared_memory
import numpy as np
from PIL import Image
import io
import base64
from redis import Redis
import json
import time
from tasks import postprocess

r = Redis(host="inference-redis", port="6379", decode_responses=True)
BATCH_SIZE = 8

def get_batch(model_names):
    batches = [r.zrange(f"infer:{m}", 0, BATCH_SIZE - 1, withscores=True) for m in model_names]
    now = time.time()
    average_ages = [np.mean([float(b[1]) - now for b in batch]) for batch in batches]
    lengths = [len(batch) / BATCH_SIZE for batch in batches]
    fitnesses = [age / 30 + length for age, length in zip(average_ages, lengths)]
    model_index = fitnesses.index(max(fitnesses))
    batch = batches[model_index]
    selected_model = model_names[model_index]
    print("----------------------------------MODEL--------------------------------------")
    print(selected_model)
    r.zrem(f"infer:{selected_model}", *[b[0] for b in batch])
    r.hincrby(f"requests", selected_model, -len(batch))
    batch = [json.loads(b[0]) for b in batch]
    print("----------------------------------BATCH--------------------------------------")
    print(len(batch))
    return batch

def infer_loop():
    while True:
        time.sleep(0.1)
        request_counts = r.hgetall("requests")
        model_names = [model_name for model_name, count in request_counts.items() if int(count) > 0]
        if not model_names:
            continue
        print("--------------------------------REQUEST_COUNTS_BEFORE--------------------------------------")
        print(request_counts)
        batch = get_batch(model_names)
        print("--------------------------------REQUEST_COUNTS_AFTER--------------------------------------")
        print(r.hgetall("requests"))
        print("--------------------------------INFERRING--------------------------------------")
        images = []
        for b in batch:
            print(b["id"])
            images.append(load_image(b))
        time.sleep(0.1)
        for index, im in enumerate(images):
            info = write_response(im)
            postprocess.s(info, batch[index]["id"]).delay()
        
def load_image(args):
    shm = shared_memory.SharedMemory(name=args["chunk_name"])
    image = np.ndarray(args["image_shape"], dtype=args["image_dtype"], buffer=shm.buf)
    image = Image.fromarray(image).resize((640, 640))
    im_arr = np.asarray(image)
    im2 = np.ndarray(args["image_shape"], dtype=args["image_dtype"])
    im2[:] = im_arr
    shm.close()
    shm.unlink()
    return im2

def write_response(im_arr):
    shm2 = shared_memory.SharedMemory(create=True, size=im_arr.nbytes)
    shared = np.ndarray(im_arr.shape, dtype=im_arr.dtype, buffer=shm2.buf)
    shared[:] = im_arr[:]
    return_val = {"chunk_name": shm2.name, "image_shape": im_arr.shape, "image_dtype": im_arr.dtype.name}
    shm2.close()
    return return_val


if __name__ == "__main__":
    infer_loop()