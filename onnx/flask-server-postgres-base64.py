from flask import Flask, request, jsonify
from yolov8 import YOLOv8
import cv2
import numpy as np
import urllib.request
import psycopg2
import json
import os
import datetime
import base64


app = Flask(__name__)
db_init_args = {
    "database": "postgres",
    "host": os.environ.get("AWS_POSTGRES_DB_HOST"),
    "user": os.environ.get("AWS_POSTGRES_DB_USER"),
    "password": os.environ.get("AWS_POSTGRES_DB_PASSWORD"),
    "port": 5432,
}
conn = psycopg2.connect(**db_init_args)

class_names = {0: 'adult', 1: 'nipple', 2: 'underage'}
yolov8_detector = YOLOv8("model.onnx", conf_thres=0.1, iou_thres=0.1)


def get_user_token_by_token(token, cursor):
    cursor.execute("select * from api_tokens where token = %s;", (token,))
    return cursor.fetchone()


def get_user_by_id(user_id, cursor):
    cursor.execute("select * from users where id = %s;", (user_id,))
    return cursor.fetchone()


def decode_base64_image(encoded_image):
    image_data = base64.b64decode(encoded_image)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


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


def predict_single_image(base64_image):
    img = decode_base64_image(base64_image)

    img = resize_with_pad(img, (800, 800))
    _, scores, class_ids = yolov8_detector(img)
    prediction = map(
        lambda x: [class_names[x[0]], float(x[1])], zip(class_ids, scores)
    )
    return list(prediction)


def create_formatted_response(predictions):
    nipple_scores = [x[1] for x in predictions if x[0] == "nipple"]
    child_scores = [x[1] for x in predictions if x[0] == "underage"]
    adult_scores = [x[1] for x in predictions if x[0] == "adult"]

    nipple_max_score = max(nipple_scores) if len(nipple_scores) > 0 else 0
    child_max_score = max(child_scores) if len(child_scores) > 0 else 0
    adult_max_score = max(adult_scores) if len(adult_scores) > 0 else 0

    child_nsfw_score = 0
    if child_max_score > 0 and nipple_max_score > 0:
        child_nsfw_score = max(child_max_score, nipple_max_score)

    adult_nsfw_score = 0
    if adult_max_score > 0 and nipple_max_score > 0:
        adult_nsfw_score = max(adult_max_score, nipple_max_score)

    formatted_response = {
        "categories": {
            "child_nsfw": True if child_nsfw_score > 0 else False,
            "adult_nsfw": True if adult_nsfw_score > 0 else False,
        },
        "scores": {
            "child_nsfw": child_nsfw_score,
            "adult_nsfw": adult_nsfw_score,
        },
    }
    return formatted_response


@app.route("/v0/classify/image", methods=["POST"])
def home():
    data = request.get_json()
    base64_image = data.get("input")

    if not base64_image:
        return jsonify({"result": "base64_image not provided"}), 400
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"result": "No token"}), 401
    
    cursor = conn.cursor()

    user_token = get_user_token_by_token(token, cursor)
    if not user_token:
        cursor.close()
        return jsonify({"result": "Invalid token"}), 401
    
    user_id = user_token[1]
    user = get_user_by_id(user_id, cursor)
    if not user:
        cursor.close()
        return jsonify({"result": "User not found"}), 404
    
    balance = user[12]
    if balance <= 0:
        cursor.close()
        return jsonify({"result": "No queries left"}), 402

    prediction = predict_single_image(base64_image)
    cursor.close()

    formatted_response = create_formatted_response(prediction)

    if data.get("return_version"):
        formatted_response["version"] = "1.1.0"

    return jsonify(formatted_response), 200


@app.route("/", methods=["GET"])
def home_get():
    return "Hello World!"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=80)
    # fastwsgi.run(wsgi_app=app, host='0.0.0.0', port=80)
    # app.run(debug=True, host="0.0.0.0", port=80, threaded=True)
    # app.run(debug=True, host="0.0.0.0", port=80, threaded=False, processes=64)
