import argparse
import os
import yaml
import sys
from typing import Any, List, Union, Dict
from itertools import product
from copy import deepcopy
import requests
import cv2
import numpy as np
import base64
from inference.models.utils import get_roboflow_model
from inference.core.registries.roboflow import get_model_type
from inference.core.managers.metrics import get_system_info
from inference.core.version import __version__
from time import perf_counter
import string
import random
import json


args = {
    "api_key": (None, "API key for Roboflow inference project or core model ID"),
    "batch_size": ([1], "Batch size for inference"),
    "confidence": ([0.5], "Confidence threshold for inference"),
    "config": (None, "Path to YAML config file"),
    "gpu_id": (
        0,
        "GPU ID used to automatically populate GPU info in results for pip inference (this arg will not set the GPU to be used by the inference package)",
    ),
    "http_base_url": ("http://localhost:9001", "Base URL for HTTP inference"),
    "image": (None, "Image to run inference on (can be local path or URL)"),
    "image_size": (None, "Image size before preprocessing"),
    "interface": (["pip"], "Interface for benchmarking (pip or http)"),
    "iou_threshold": ([0.5], "IOU threshold for inference"),
    "model_id": (None, "Model ID for Roboflow inference project or core model ID"),
    "num_iterations": (100, "Number of iterations to run inference"),
    "num_warmup_iterations": (10, "Number of warmup iterations to run inference"),
    "output_directory": (None, "Directory to save benchmark results"),
}

filter_result_keys = [
    "api_key",
    "config",
    "gpu_id",
    "http_base_url",
    "image",
]


def parse_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Warning: Error parsing YAML file, ignoring: {exc}")
            return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmarking for Roboflow inference project."
    )
    for arg_name, (arg_default, arg_help) in args.items():
        add_argument_args = {
            "help": arg_help,
        }

        if isinstance(arg_default, list):
            add_argument_args["nargs"] = "+"
            add_argument_args["type"] = type(arg_default[0])
        else:
            add_argument_args["type"] = (
                type(arg_default) if arg_default is not None else str
            )

        parser.add_argument(f"--{arg_name}", **add_argument_args)

    known, unknown = parser.parse_known_args(sys.argv[1:])
    if len(unknown) > 0:
        print(f"Warning: Ignoring unknown args: {unknown}")

    if known.config is not None and os.path.exists(known.config):
        config = parse_yaml(known.config)
        print("CONFIG", config)
    else:
        config = {}

    final_args = {
        arg_name: getattr(known, arg_name)
        or config.get(
            arg_name,
            arg_default if not isinstance(arg_default, list) else arg_default[0],
        )
        for arg_name, (arg_default, arg_help) in args.items()
    }
    return final_args


def unwrap_config(config: Dict[str, Union[str, List[str]]]) -> List[Dict[str, str]]:
    keys_with_list_values = {k: v for k, v in config.items() if isinstance(v, list)}
    keys_with_non_list_values = {
        k: v for k, v in config.items() if not isinstance(v, list)
    }

    if not keys_with_list_values:  # Base case: all values are non-lists.
        return [keys_with_non_list_values]

    unwrapped_configs = []
    for combination in product(*keys_with_list_values.values()):
        combined_config = {
            **keys_with_non_list_values,
            **dict(zip(keys_with_list_values.keys(), combination)),
        }
        unwrapped_configs.append(combined_config)

    return unwrapped_configs


def benchmark_with_config(config: Dict[str, Any], model, task):
    results = deepcopy(config)
    if "http" in results.get("image"):
        response = requests.get(results.get("image"), stream=True)
        image_np = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(results.get("image"), cv2.IMREAD_COLOR)
    if results.get("image_size") is not None:
        image = cv2.resize(
            image,
            (int(results.get("image_size")), int(results.get("image_size"))),
            interpolation=cv2.INTER_AREA,
        )
        results["image_size"] = (results.get("image_size"), results.get("image_size"))
    else:
        results["image_size"] = image.shape[0:2]

    results["model_input_size"] = (model.img_size_h, model.img_size_w)
    results["task"] = model.task_type
    if results.get("interface") == "pip":

        def inference_fn():
            res = model.infer(
                image,
                confidence=results.get("confidence"),
                iou_threshold=results.get("iou_threshold"),
            )
            num_detections = len(res[0])
            return num_detections

        system_info = get_system_info()
        if system_info["gpu_count"] > 0:
            if len(system_info["gpus"]) >= results.get("gpu_id"):
                system_info["gpu_name"] = system_info["gpus"][results.get("gpu_id")][
                    "name"
                ]
                system_info["gpu_memory"] = system_info["gpus"][results.get("gpu_id")][
                    "total_memory"
                ]
        system_info = {
            k: v
            for k, v in system_info.items()
            if k
            in [
                "platform",
                "platform_release",
                "platform_version",
                "architecture",
                "processor",
                "gpu_name",
                "gpu_memory",
            ]
        }

    else:
        _, img_encoded = cv2.imencode(".jpg", image)
        base64_image = base64.b64encode(img_encoded).decode("utf-8")
        image_val = {"type": "base64", "value": base64_image}
        if results.get("batch_size") > 1:
            image_val = [image_val] * results.get("batch_size")
        payload = {
            "image": image_val,
            "confidence": results.get("confidence"),
            "iou_threshold": results.get("iou_threshold"),
            "model_id": results.get("model_id"),
            "api_key": results.get("api_key"),
        }

        url = f"{results.get('http_base_url')}/infer/{task.replace('-','_')}?api_key={results.get('api_key')}"
        headers = {"Content-Type": "application/json"}

        def inference_fn():
            res = requests.post(url, json=payload, headers=headers)
            try:
                res.raise_for_status()
                num_detections = len(res.json()["predictions"])
                return num_detections
            except:
                return 0

        try:
            import docker

            client = docker.from_env()
            containers = client.containers.list()
            image = get_roboflow_container_image(containers)
            results["roboflow_inference_server_image"] = image
        except:
            pass

    times = []
    fps = None
    rps = None
    for i in range(results.get("num_iterations")):
        start = perf_counter()
        num_preds = inference_fn()
        end = perf_counter()
        if i >= results.get("num_warmup_iterations"):
            times.append(end - start)
        if i % 10 == 0:
            rps = 1 / np.mean(times)
            fps = rps * results.get("batch_size")
        print_over_line(
            f"{i}/{results.get('num_iterations')} | RPS: {rps:.1f} | FPS: {fps:.1f}"
        )
    results["num_predictions"] = num_preds
    rps = 1 / np.mean(times)
    fps = rps * results.get("batch_size")
    print_over_line(f"FPS: {fps:.1f} for {results}")
    results["rps"] = rps
    results["fps"] = fps
    results = {k: v for k, v in results.items() if k not in filter_result_keys}
    if results.get("output_directory") is not None:
        if not os.path.exists(results.get("output_directory")):
            os.makedirs(results.get("output_directory"))
        output_file = os.path.join(
            results.get("output_directory"), f"{random_string()}_{__version__}.json"
        )
        with open(output_file, "w") as f:
            f.write(json.dumps(results, indent=4))


def random_string(length=10):
    letters = string.ascii_lowercase
    numbers = string.digits
    characters = letters + numbers
    return "".join(random.choice(characters) for i in range(length))


def print_over_line(*args):
    print(" " * 80, end="\r")
    print(*args, end="\r")


def get_roboflow_container_image(containers):
    for c in containers:
        if "roboflow/roboflow-inference-server" in c.attrs["Config"]["Image"]:
            return c.attrs["Config"]["Image"]


def main():
    args = parse_args()
    assert args["image"] is not None
    model = get_roboflow_model(
        api_key=args.get("api_key"),
        model_id=args.get("model_id"),
    )
    task, model_type = get_model_type(args.get("model_id"), api_key=args.get("api_key"))
    args["model_type"] = model_type
    configs = unwrap_config(args)
    for config in configs:
        benchmark_with_config(config, model, task)


if __name__ == "__main__":
    main()
