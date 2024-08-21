import os
import json
from PIL import Image


class COCOFormatMaker():
    """
    Create annotation in COCO format just by dir with images and boxes
    """
    def __init__(self, train_path: str, val_path: str = None):
        self.train = train_path
        self.val = val_path
        self.categ = []
        self.data = {"images": [], "annotations": [], "categories": self.categ}

    def denormalize_bbox(self, bbox, image_w, image_h):
        x_center, y_center, w, h = bbox
        w = w * image_w
        h = h * image_h
        x1 = ((2 * x_center * image_w) - w) / 2
        y1 = ((2 * y_center * image_h) - h) / 2
        return [round(x1), round(y1), round(w), round(h)]

    def categories(self):
        with open('/datasets/traffic_sign_pothole/label_map.json', 'r') as f:
            categ = json.load(f)
        for key in categ.keys():
            d = {}
            d['id'] = int(categ[key])
            d['name'] = key
            self.categ.append(d)

    def img_info(self, name: str, img_id: int, img_path: str):
        data = {}
        data['id'] = img_id
        data['file_name'] = name
        with Image.open(img_path+'images/'+name) as img:
            width, height = img.size
        data['height'] = height
        data['width'] = width

        return data, height, width

    def annotation_inf(self, file_data: str, img_id: int, ann_id: int, h, w):
        data = {}
        data['id'] = ann_id
        data['image_id'] = img_id

        info = file_data.split(' ')

        # because start from 1
        data['category_id'] = int(info[0]) + 1
        bbox = self.denormalize_bbox([float(i) for i in info[1:]], h, w)
        data['bbox'] = bbox
        data['iscrowd'] = 0
        data['area'] = data['bbox'][2] * data['bbox'][3]
        data['segmentation'] = []

        return data

    def merge_different_annos(self, anno_paths: list, save_name: str, save_dir: str):
        """
        Merge some difference annotation in COCO-format. Save result anno
        """
        busy_anno_ids = []

        for i, anno in enumerate(anno_paths):
            with open(anno, 'r') as f:
                data = json.load(f)
            if i == 0:
                self.data = data
                for a in data['annotations']:
                    busy_anno_ids.append(a['id'])
                ids_from = max(busy_anno_ids) + 1
            else:
                ann_list = data['annotations']
                categ = data['categories']
                images = data['images']

                # ATTENTION! It is believed that 'id' should not intersect
                for c in categ:
                    self.data['categories'].append(c)

                for img in images:
                    self.data['images'].append(img)
                    for an in ann_list:
                        if an['image_id'] == img["id"]:
                            new_ann = an.copy()
                            new_ann["id"] = ids_from
                            ids_from += 1
                            self.data['annotations'].append(new_ann)

        with open(os.path.join(save_dir, f'{save_name}.json'), 'w') as f:
            json.dump(self.data, f)


    def from_yolo_to_coco(self, save_name, input_dir, output_dir, categ=[{"id": 156, "name": "pothole"}]):
        # Define the categories for the COCO dataset
        categories = categ

        # Define the COCO dataset dictionary
        coco_dataset = {
            "info": {},
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": []
        }

        loss = 0
        idx = 10000000

        # Loop through the images in the input directory
        for image_file in os.listdir(input_dir):

            if image_file[-4:] == '.txt' or image_file == '.DS_Store':
                continue

            # Load the image and get its dimensions
            image_path = os.path.join(input_dir, image_file)
            image = Image.open(image_path)
            width, height = image.size

            idx += 1

            # Add the image to the COCO dataset
            image_dict = {
                "id": idx,
                "width": width,
                "height": height,
                "file_name": image_file
            }

            # Load the bounding box annotations for the image
            if '(' in image_file:
                image_file = image_file[:-7] + '.jpg'

            try:
                with open(os.path.join(input_dir, f'{image_file[:-4]}.txt')) as f:
                    annotations = f.readlines()
            except:
                idx -= 1
                loss += 1

                print(f'Loss: {loss}')
                continue

            coco_dataset["images"].append(image_dict)

            # Loop through the annotations and add them to the COCO dataset
            for ann in annotations:
                x, y, w, h = map(float, ann.strip().split()[1:])
                x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
                x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
                ann_dict = {
                    "id": len(coco_dataset["annotations"]),
                    "image_id": idx,
                    "category_id": 156,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0
                }
                coco_dataset["annotations"].append(ann_dict)

        # Save the COCO dataset to a JSON file
        with open(os.path.join(output_dir, f'{save_name}.json'), 'w') as f:
            json.dump(coco_dataset, f)

    def main(self, dirs_yolo, save_path):
        anno_to_merge = [
                    ['/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign_pothole/train_anno_traffic_sign.json',
                     '/Users/alina/PycharmProjects/obj_detection/train_anno_pothole.json'],
            ['/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign_pothole/val_anno_traffic_sign.json',
             '/Users/alina/PycharmProjects/obj_detection/test_anno_pothole.json']
        ]
        names_pothole = ['train_anno_pothole', 'test_anno_pothole']
        res_names = ['train_anno_custom', 'test_anno_custom']

        for i, dir in enumerate(dirs_yolo):
            creator.from_yolo_to_coco(names_pothole[i], dir, save_path)

        for i, dirs in enumerate(anno_to_merge):
            self.merge_different_annos(dirs, res_names[i], save_path)


if __name__ == "__main__":
    creator = COCOFormatMaker(
        "/datasets/traffic_sign_pothole/dataset/train/",
        "/datasets/traffic_sign_pothole/dataset/valid/"
    )
    dirs_yolo = [
        '/Users/alina/Desktop/pothole_dataset_v8 2/all/train/',
        '/Users/alina/Desktop/pothole_dataset_v8 2/all/test/'
    ]
    save_path = "/"
    creator.main(dirs_yolo, save_path)