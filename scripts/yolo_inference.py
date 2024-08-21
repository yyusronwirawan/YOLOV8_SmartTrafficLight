import cv2
import numpy as np
import yaml
import os
from ultralytics import YOLO
import onnx

# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2.dnn
import numpy as np

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml("coco8.yaml"))["names"]
CONFIDENCE, SCORE_THRESHOLD, IOU_THRESHOLD = 0.5, 0.5, 0.5
CONF_PATH = "/conf/main_conf.yaml"
font_scale, thickness = 1, 1

class Predictor:
    def __init__(self):
        with open(CONF_PATH, "r") as yamlfile:
            main_conf = yaml.load(yamlfile, Loader=yaml.FullLoader)

        self.processed_path = main_conf["processed_path"]

        self.labels_path = main_conf["labels"]
        self.labels = open(self.labels_path).read().strip().split("\n")
        self.path_data_dir = main_conf["dataset"]
        self.dataset = os.listdir(main_conf["dataset"])
        self.colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        self.model = YOLO('/train/runs/detect/yolov8n_custom/weights/best.onnx')
        onnx_path = self.model.export(format="onnx")
        onnx.checker.check_model(onnx_path)
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_path)


    def get_img(self, img_name: str = None, cap=None) -> np.array:
        if cap:
            ret, image = cap.read()
            if not(ret):
                return [False]*4
        else:
            image = cv2.imread(self.path_data_dir + "/" + img_name)
        return image

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        """
        Draws bounding boxes on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the bounding box on.
            class_id (int): Class ID of the detected object.
            confidence (float): Confidence score of the detected object.
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        """
        label = f"{CLASSES[class_id]} ({confidence:.2f})"
        color = self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(
            img,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
        )

    def make_prediction(self, original_image):
        """
        Main function to load ONNX train, perform inference, draw bounding boxes, and display the output image.

        Args:
            onnx_model (str): Path to the ONNX train.
            input_image (str): Path to the input image.

        Returns:
            list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
        """
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        scale = length / 640

        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        self.model.setInput(blob)

        # Perform inference
        outputs = self.model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= CONFIDENCE:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            self.draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        # # Display the image with bounding boxes
        # cv2.imshow("image", original_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return original_image

    def val(self, samples=1, video_file: str = False):
        if not video_file:
            for i in range(samples):
                path = self.dataset[i]
                img = self.get_img(path)
                res_img = self.make_prediction(
                    img
                )
                answer = cv2.imwrite(
                    self.processed_path + f"result_example_{i}.jpg", res_img
                )
                if answer:
                    print("Image saved successfully")
                else:
                    print("Unable to save image")
        else:
            cap = cv2.VideoCapture(video_file)
            _, image = cap.read()
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            out = cv2.VideoWriter(
                "/Users/alina/PycharmProjects/obj_detection/data/video/output.avi",
                fourcc,
                20.0,
                (w, h),
            )

            cnt = 0
            while True:
                cnt += 1

                img = self.get_img(cap=cap)
                if type(img[0]) is bool:
                    break

                res_img = self.make_prediction(
                    img
                )
                out.write(res_img)
                # cv2.imshow("image", res_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()


def run_on_images(yolo: Predictor, samples: int) -> None:
    yolo.val(samples=samples)


def run_on_video(yolo: Predictor, video_path: str):
    yolo.val(video_file=video_path)


if __name__ == "__main__":
    yolo = Predictor()
    run_on_images(15)

