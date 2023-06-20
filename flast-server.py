import json
import json
from flask import Flask, request, jsonify
import logging
from ultralytics import YOLO
import urllib.request
import cv2
import numpy as np


model = YOLO("yolov8_custom_trained_model.pt")  # load a pretrained model (recommended for training)


app = Flask(__name__)
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

class_names = {0: 'adult', 1: 'nipple', 2: 'underage'}

def get_image_from_url(image_url):
    try:
        req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as url:
            image_contents = url.read()
    except urllib.error.HTTPError as e:
        return {"error": str(e)}
    except ConnectionResetError:
        return {"error": "ConnectionResetError"}

    arr = np.asarray(bytearray(image_contents), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return {"image": img}


def predict_single_image(image_url):
    image = get_image_from_url(image_url)
    if "error" in image:
        return image
    while True:
        try:
            results = model(image["image"])
            break
        except TypeError:
            print("Retrying...")
            continue
        
    processed_resutls = []

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        for box in boxes:
            confidence = box.conf.item()
            name = class_names[int(box.cls.item())]
            processed_resutls.append([name, confidence])

    return processed_resutls


@app.route("/", methods=["POST"])
def home():
    data = request.get_json()
    image_url = data.get("image_url")

    # download image


    if not image_url:
        return jsonify({"result": "No image_url"}), 400
    token = request.headers.get("Authorization")
    if token != "8Jw1kj5Woa4SDHtD%hH!7v%NBox0!47^WL4--":
        return jsonify({"result": "Invalid token"}), 401
    print(image_url)
    result = {
        "predictions": predict_single_image(image_url),
    }

    if data.get("return_version"):
        result["version"] = "1.0.5"

    return jsonify(result), 200


@app.route("/", methods=["GET"])
def home_get():
    return "Hello World!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
    # app.run(debug=True, host="0.0.0.0", port=80, threaded=False, processes=64)
    # app.run(debug=True, host="0.0.0.0", port=80, threaded=False)