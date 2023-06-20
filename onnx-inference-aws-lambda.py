import json
from imread_from_url import imread_from_url
from yolov8 import YOLOv8
import cv2

class_names = {0: 'adult', 1: 'nipple', 2: 'underage'}
yolov8_detector = YOLOv8("yolov8_custom_trained_model.onnx", conf_thres=0.1, iou_thres=0.1)


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
    img = imread_from_url(img_url)
    img = resize_with_pad(img, (800, 800))
    _, scores, class_ids = yolov8_detector(img)
    prediction = map(
        lambda x: [class_names[x[0]], float(x[1])], zip(class_ids, scores)
    )
    return list(prediction)


def make_return(status_code, result, return_version=False):
    body = {
        "result": result,
    }
    if return_version:
        body["version"] = "1.0.4"
    return body


def handler(event, context):
    body = json.loads(event.get("body", "{}"))

    image_url = body.get("image_url")
    return_version = body.get("return_version")

    token = event.get("headers", {}).get("authorization")

    if not image_url:
        return make_return(400, "No image_url", return_version)
    if token != "8Jw1kj5Woa4SDHtD%hH!7v%NBox0!47^WL4--":
        return make_return(401, "Invalid token", return_version)

    print(image_url)

    return make_return(200, predict_single_image(image_url), return_version)