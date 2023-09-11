import base64
import requests
from multiprocessing import Pool
import asyncio
import aiohttp


def encode_bas64(image_path):
    with open(image_path, "rb") as image:
        image_string = base64.b64encode(image.read())

    return image_string.decode("ascii")


def request1(i):
    print(i)
    r = requests.post(
        "http://localhost:8000/infer",
        json={"image": encode_bas64("testim.jpg"), "model": "model"},
    )
    print(r.text)
    print(f"FINISHED {i}")
    if r.status_code == 200:
        return 1
    return 0


def request2(i):
    r = requests.post(
        "http://localhost:8000/infer",
        json={"image": encode_bas64("testim.jpg"), "model": "model2"},
    )
    print(r.text)
    print(f"FINISHED {i}")
    if r.status_code == 200:
        return 1
    return 0

def request3(i):
    print(i)
    print(requests.get("http://localhost:8000/").text)
    print(f"FINISHED {i}")

async def do_request(session, i):
    print(f"Starting {i}")
    async with session.post(
        'http://localhost:8000/infer',
        json={"image": encode_bas64("testim.jpg"), "model": "model2"},
    ) as response:
        resp = await response.json()
        print(f"Finished {i}")
        print(resp)
        return resp

async def main():
    tasks = []
    async with aiohttp.ClientSession(read_timeout=0) as session:
        for i in range(50000):
            tasks.append(do_request(session, i))
        await asyncio.gather(*tasks)
    

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())