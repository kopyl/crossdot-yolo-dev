import time
init_time = time.time()
from flask import Flask, request, jsonify
from yolov8 import YOLOv8
import cv2
import numpy as np
import urllib.request
import uuid
import os
import boto3
import json


app = Flask(__name__)
aws_session = boto3.Session(
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
)
dynamodb = aws_session.resource('dynamodb', region_name='us-east-1')
users_table = dynamodb.Table('users')
queries_table = dynamodb.Table('queries')

class_names = {0: 'adult', 1: 'nipple', 2: 'underage'}
yolov8_detector = YOLOv8("model.onnx", conf_thres=0.1, iou_thres=0.1)


def save_query_json(user_id, image_url, prediction):
    query = {
        "image_url": image_url,
        "prediction": prediction
    }
    query_json = json.dumps(query)
    queries_table.put_item(
        Item={
            "query_id": str(uuid.uuid4()),
            "user_id": user_id,
            "request_json": query_json,
            "date_created": int(time.time())
        }
    )


def get_user(user_id):
    response = users_table.get_item(
        Key={
            'user_id': user_id
        }
    )
    return response.get("Item")


def reduce_queries_left(user_id):
    response = users_table.update_item(
        Key={
            'user_id': user_id
        },
        UpdateExpression="set queries_left = queries_left - :val",
        ExpressionAttributeValues={
            ':val': 1
        },
        ReturnValues="UPDATED_NEW"
    )
    return response.get("Attributes").get("queries_left")


def get_image_from_url(image_url):
    try:
        req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as url:
            image_contents = url.read()
    except urllib.error.HTTPError as e:
        return {"error": "HTTPError", "reason": "Was not able to download image."}
    except urllib.error.URLError as e:
        return {"error": "URLError", "reason": "Was not able to download image."}
    except ConnectionResetError:
        return {"error": "ConnectionResetError", "reason": "Was not able to download image."}

    arr = np.asarray(bytearray(image_contents), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return {"image": img}


print("Init time: ", time.time() - init_time)


def resize_with_pad(image,
                    new_shape,
                    padding_color = (255, 255, 255)):
    """
    https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


def predict_single_image(img_url):
    img = get_image_from_url(img_url)
    if "error" in img:
        return img
    img = img["image"]

    img = resize_with_pad(img, (800, 800))
    _, scores, class_ids = yolov8_detector(img)
    prediction = map(
        lambda x: [class_names[x[0]], float(x[1])], zip(class_ids, scores)
    )
    return list(prediction)


@app.route("/", methods=["POST"])
def home():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"result": "No image_url"}), 400
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"result": "No token"}), 401

    user = get_user(token)
    if not user:
        return jsonify({"result": "Invalid user"}), 401
    queries_left = user.get("queries_left")
    if queries_left <= 0:
        return jsonify({"result": "No queries left"}), 402

    print(image_url)

    prediction = predict_single_image(image_url)
    result = {
        "predictions": prediction,
    }

    save_query_json(token, image_url, prediction)
    reduce_queries_left(token)

    if data.get("return_version"):
        result["version"] = "1.0.5"

    return jsonify(result), 200


@app.route("/", methods=["GET"])
def home_get():
    return "Hello World!"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=80)
    # app.run(debug=True, host="0.0.0.0", port=80, threaded=False, processes=64)
