import base64
import requests
from multiprocessing import Pool


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


if __name__ == "__main__":
    with Pool(processes=50) as pool:
        # pool.map(request3, range(50))
        dones = pool.map(request2, range(10))
        dones.extend(pool.map(request1, range(50)))
        print(sum(dones))
