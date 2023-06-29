from flask import Flask, request, jsonify
from yolov8 import YOLOv8
import cv2
import numpy as np
import urllib.request
import psycopg2
import json
import os
import datetime


app = Flask(__name__)
db_init_args = {
    "database": "postgres",
    "host": os.environ.get("AWS_POSTGRES_DB_HOST"),
    "user": os.environ.get("AWS_POSTGRES_DB_USER"),
    "password": os.environ.get("AWS_POSTGRES_DB_PASSWORD"),
    "port": 5432,
}
conn = psycopg2.connect(**db_init_args)

class_names = {
    0: 'adult',
    1: 'exposed_anus',
    2: 'exposed_buttock',
    3: 'female_genitalia',
    4: 'male_genitalia',
    5: 'nipple',
    6: 'underage'
}
model_path = os.environ.get("ONNX_MODEL_PATH", "model.onnx")
yolov8_detector = YOLOv8(model_path, conf_thres=0.35, iou_thres=0.6)


def get_user_token_by_token(token, cursor):
    cursor.execute("select * from api_tokens where token = %s;", (token,))
    return cursor.fetchone()


def get_user_by_id(user_id, cursor):
    cursor.execute("select * from users where id = %s;", (user_id,))
    return cursor.fetchone()


def save_query(image_url, detected_objects, user_id, cursor):
    created_at = datetime.datetime.now()
    updated_at = created_at
    cursor.execute(
        "insert into generated_images "
        "(created_at, updated_at, image_url, detected_objects, user_id) "
        "values (%s, %s, %s, %s, %s);",
        (created_at, updated_at, image_url, json.dumps(detected_objects), user_id)
    )


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
    boxes, scores, class_ids = yolov8_detector(img)
    prediction = map(
        lambda x:
            {
                "label": class_names[x[0]],
                "score": float(x[1]),
                "box": [float(x) for x in x[2]]
            },
        zip(class_ids, scores, boxes)
    )
    return list(prediction)


def create_formatted_response(predictions):
    def get_max_score(category):
        scores = [x["score"] for x in predictions if x["label"] == category]
        return max(scores) if len(scores) > 0 else 0

    child_max_score = get_max_score("underage")

    nsfw_max_score = max([
        get_max_score("nipple"),
        get_max_score("exposed_anus"),
        get_max_score("exposed_buttock"),
        get_max_score("female_genitalia"),
        get_max_score("male_genitalia"),
    ])

    child_nsfw_score = 0
    if child_max_score > 0 and nsfw_max_score > 0:
        child_nsfw_score = max(child_max_score, nsfw_max_score)

    other_nsfw_score = 0
    if nsfw_max_score > 0 and not child_nsfw_score:
        other_nsfw_score = nsfw_max_score

    formatted_response = {
        "categories": {
            "child_nsfw": child_nsfw_score > 0,
            "other_nsfw": other_nsfw_score > 0,
        },
        "confidence_scores": {
            "child_nsfw": child_nsfw_score,
            "other_nsfw": other_nsfw_score,
        },
        "predictions": predictions,
    }
    return formatted_response


@app.route("/v0/classify/image", methods=["POST"])
def home():
    data = request.get_json()
    image_url = data.get("input")

    if not image_url:
        return jsonify({"result": "Image url not provided"}), 400
    print(image_url)
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

    new_balance = balance - 0.0003
    cursor.execute("update users set balances = %s where id = %s;", (new_balance, user_id))

    prediction = predict_single_image(image_url)
    save_query(image_url, prediction, user_id, cursor)

    conn.commit()
    cursor.close()

    formatted_response = create_formatted_response(prediction)

    if data.get("return_version"):
        formatted_response["version"] = "1.3.2"

    return jsonify(formatted_response), 200


@app.route("/", methods=["GET"])
def home_get():
    return "Hello World!"


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=80)
    # fastwsgi.run(wsgi_app=app, host='0.0.0.0', port=80)
    # app.run(debug=True, host="0.0.0.0", port=80, threaded=True)
    # app.run(debug=True, host="0.0.0.0", port=80, threaded=False, processes=64)
