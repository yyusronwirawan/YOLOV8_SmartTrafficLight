from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
import os, json, cv2, random
from detectron2.engine import DefaultTrainer
import pickle
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


PROCESSED_IMG = '/Users/alina/PycharmProjects/obj_detection/data/processed_img/'


def create_conf(is_trained):
    epoch = 5
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("traffic_sign_train_4",)
    cfg.DATASETS.TEST = ("traffic_sign_test_4",)
    cfg.DATALOADER.NUM_WORKERS = 2
    if is_trained:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    # It is not epochs
    cfg.SOLVER.MAX_ITER = 60000 / cfg.SOLVER.IMS_PER_BATCH * epoch
    cfg.MODEL.DEVICE = "cpu"

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 156
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

    return cfg


def register_dataset():
    register_coco_instances(
        f"traffic_sign_train", {},
        f"/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset/train/images/train_anno_custom.json",
        "/Users/alina/Desktop/traffic_sign_dataset_raw/rtsd-frames/rtsd-frames"
    )
    register_coco_instances(
        f"traffic_sign_test", {},
        f"/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset/valid/images/test_anno_custom.json",
        "/Users/alina/Desktop/traffic_sign_dataset_raw/rtsd-frames/rtsd-frames"
    )


def train():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    with open("/Users/alina/PycharmProjects/obj_detection/models/detectron2/cfg.pkl", "wb") as f:
        pickle.dump(cfg, f)

    return cfg


def predict(cfg):
    my_dataset_test_metadata = MetadataCatalog.get("traffic_sign_test")
    dataset_dicts = DatasetCatalog.get("traffic_sign_test")

    # means confidence
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    predictor = DefaultPredictor(cfg)

    cnt = 1

    for d in random.sample(dataset_dicts, 2):
        cnt += 1
        im = cv2.imread(d["file_name"])
        outputs = predictor(im[:, :, ::-1])
        print(outputs)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        v = Visualizer(im[:, :, ::-1], metadata=my_dataset_test_metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(PROCESSED_IMG+f'detectron{cnt}.jpg', out)


def evaluate(cfg, trainer, predictor):
    evaluator = COCOEvaluator("traffic_sign_test", cfg, False, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "traffic_sign_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == '__main__':
    cfg = create_conf()
    my_dataset_train_metadata, dataset_dicts_train = register_dataset()

    image_folder = "/Users/alina/Desktop/traffic_sign_dataset_raw/rtsd-frames/rtsd-frames"
    output_folder = "/Users/alina/PycharmProjects/obj_detection/data/from_detectron2/"
    model_weights = "/Users/alina/PycharmProjects/obj_detection/train/output/model_final.pth"

    predict(cfg, model_weights, image_folder, output_folder)