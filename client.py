import requests

dataset_id = "snakes-inst-seg"
version_id = "4"
image_url = "https://www.saferbrand.com/media/Articles/Safer-Brand/identify-and-get-rid-of-venmous-snakes.jpg"
image_url2= "https://source.roboflow.one/AWklLvEzMUUAhCA8Jl82sBDxIwd2/UVxRzfzxtReSjo5myY1l/original.jpg"
image_url = "https://www.usanetwork.com/sites/usablog/files/styles/scale_1280/public/2022/07/snake-in-the-grass-encounter_0.jpg"
#Replace ROBOFLOW_API_KEY with your Roboflow API Key
api_key = "rf_AWklLvEzMUUAhCA8Jl82sBDxIwd2"
api_key = "GYaBMWQ6xDqVFsEJIoan"
confidence = 0.5

url = f"http://localhost:9001/{dataset_id}/{version_id}"

import time
params = {
    "api_key": api_key,
    "confidence": confidence,
    "image": image_url2,
}

res = requests.post(url, params=params)
params = {
    "api_key": api_key,
    "confidence": confidence,
    "image": image_url,
}

start = time.time()
res = requests.post(url, params=params)
end = time.time()
print(res.json())
print(f"TOOK {end-start:.2f} SECONDS")