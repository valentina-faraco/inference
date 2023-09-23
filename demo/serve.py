from redis import Redis, ConnectionPool
from multiprocessing import shared_memory
import uuid
import json
import time
from fastapi import FastAPI, Request
import asyncio
from tasks import preprocess, decode_base64
from inference.models.utils import get_roboflow_model
import numpy as np

import logging
import threading
logging.basicConfig(level=logging.INFO)

app = FastAPI()
model = get_roboflow_model("melee/5", "Nw3QZal3hhwHP5npbWmw")

pool = ConnectionPool(host="localhost", port=6379, decode_responses=True)

@app.get("/")
async def root(request: Request):
    await asyncio.sleep(1)
    return {"message": "HI"}


TASK_RESULT_KEY = "results:{}"
TASK_STATUS_KEY = "status:{}"
FINAL_STATE = 1
INITIAL_STATE = 0

NOT_FINISHED_RESPONSE="===NOTFINISHED==="


class ResultsChecker:
    def __init__(self):
        self.tasks = []
        self.dones = dict()
        self.running = True

    def add_redis(self, r: Redis):
        self.r = r
    
    def add_task(self, t):
        self.tasks.append(t)

    def check_task(self, t):
        if t in self.dones:
            return self.dones.pop(t)
        return NOT_FINISHED_RESPONSE

    async def loop(self):
        interval = 0.1
        while self.running:
            tasks = [t for t in self.tasks]
            task_names = [TASK_STATUS_KEY.format(id_) for id_ in tasks]
            donenesses = [r.get(t) for t in task_names]
            donenesses = [int(d) for d in donenesses]
            for id_, doneness in zip(tasks, donenesses):
                if doneness == FINAL_STATE:
                    pipe = self.r.pipeline()
                    pipe.get(TASK_RESULT_KEY.format(id_))
                    pipe.delete(TASK_RESULT_KEY.format(id_))
                    pipe.delete(TASK_STATUS_KEY.format(id_))
                    result, _ , _ = pipe.execute()
                    self.tasks.remove(id_)
                    self.dones[id_] = result
            await asyncio.sleep(interval)
    
    async def wait_for_response(self, key):
        interval = 0.1
        while True:
            result = self.check_task(key)
            if result is not NOT_FINISHED_RESPONSE:
                return result
            await asyncio.sleep(interval)

checker = ResultsChecker() 

@app.on_event("startup")
async def app_startup():
    r = Redis(connection_pool=pool, decode_responses=True)
    checker.add_redis(r)
    asyncio.create_task(checker.loop())

r = Redis(connection_pool=pool, decode_responses=True)
async def wait_for_response(id_, initial_sleep=0.3, interval=0.05, timeout=float("inf")):
    start = time.time()
    await asyncio.sleep(initial_sleep)
    while time.time() < start + timeout:
        doneness = r.get(TASK_STATUS_KEY.format(id_))
        doneness = int(doneness)
        if doneness == FINAL_STATE:
            result = r.get(TASK_RESULT_KEY.format(id_))
            r.delete(TASK_RESULT_KEY.format(id_))
            r.delete(TASK_STATUS_KEY.format(id_))
            return result
        await asyncio.sleep(interval)
    raise TimeoutError

def start_task(r, id_):
    r.set(TASK_STATUS_KEY.format(id_), INITIAL_STATE)
    checker.add_task(id_)

@app.post("/infer")
async def infer(request: Request):
    r = Redis(connection_pool=pool, decode_responses=True)
    start_time = time.time()
    json_body = await request.json()
    image = json_body["image"]
    model = json_body["model"]
    id_ = str(uuid.uuid4())
    start_task(r, id_) # add to fsm queued state
    preprocess.s(model, image, id_, start_time).delay()
    results = await checker.wait_for_response(id_)
    return {"results": results}

@app.post("/infer_sync")
async def infer_sync(request: Request):
    json_body = await request.json()
    image = json_body["image"]
    image = decode_base64(image)
    image = np.asarray(image)
    results, img_dims = model.infer(image, return_image_dims=True)
    results = model.make_response(results, img_dims)
    print(results)
    return {"results": results}
# app = web.Application()
# app.add_routes([web.get('/', root),
#                 web.post('/infer', infer)])

if __name__ == "__main__":
    web.run_app(app, port=8000)
