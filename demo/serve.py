from redis import Redis
from multiprocessing import shared_memory
import uuid
import json
import time
from fastapi import FastAPI, Request
import asyncio
from tasks import preprocess

import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
async def root(request: Request):
    await asyncio.sleep(1)
    return {"message": "HI"}

r = Redis(host="inference-redis", port="6379", decode_responses=True)

TASK_RESULT_KEY = "results:{}"
TASK_STATUS_KEY = "status:{}"
FINAL_STATE = 1
INITIAL_STATE = 0

def start_task(id_):
    r.set(TASK_STATUS_KEY.format(id_), INITIAL_STATE)

async def wait_for_response(id_, interval=0.05, timeout=float("inf")):
    start = time.time()
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


@app.post("/infer")
async def infer(request: Request):
    start_time = time.time()
    logging.info("pre awai")
    json_body = await request.json()
    logging.info("pre image")
    image = json_body["image"]
    model = json_body["model"]
    logging.info("pre call")
    id_ = str(uuid.uuid4())
    start_task(id_)
    logging.info("#"*90)
    logging.info(' '*40 + f"STARTED TASK {id_}")
    logging.info("#"*90)
    preprocess.s(model, image, id_, start_time).delay()
    results = await wait_for_response(id_)
    return {"results": results}

# app = web.Application()
# app.add_routes([web.get('/', root),
#                 web.post('/infer', infer)])

if __name__ == "__main__":
    web.run_app(app, port=8000)
