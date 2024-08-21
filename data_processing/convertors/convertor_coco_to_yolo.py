import shutil
import json
import os

from tqdm.notebook import tqdm
import pandas as pd

def get_df_annotations(annotations_file, label_map_file, min_area):
    with open(annotations_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    with open(label_map_file) as f:
        label_map = json.load(f)

    id2label = {v: k for k, v in label_map.items()}

    df = pd.DataFrame(json_data['annotations'])
    df['sign_name'] = df['category_id'].map(id2label)
    df['global_group'] = df['sign_name'].apply(lambda x: x.split('_')[0])
    df = df[df['area'] >= min_area]

    return df

def get_balanced_df(df, samples_per_class):
    balanced_data = []

    for class_id in df['category_id'].unique():
        class_data = df[df['category_id'] == class_id]
        sample = class_data.sample(min(samples_per_class, len(class_data)),
                                   replace=False,
                                  random_state=42)
        balanced_data.append(sample)

        balanced_df = pd.concat(balanced_data)
        return balanced_df


def get_filter_id(coco_json_train, label_map_file, samples_per_class, min_area=0):
    df_anno = get_df_annotations(coco_json_train, label_map_file, min_area)

    print(len(df_anno['sign_name'].unique()))

    samples_per_class = 100
    balanced_df = get_balanced_df(df_anno, samples_per_class)

    return balanced_df['id'].to_list()

def convert_coco_to_yolo(coco_json, output_dir, image_dir, filter_anno=None, only_one_class=False):
    # создаем нужные папки, если их нет
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    with open(coco_json) as f:
        data = json.load(f)

    images_info = {image['id']: image for image in data['images']}

    for ann in tqdm(data['annotations']):
        if filter_anno:
            if ann['id'] not in filter_anno:
                continue

        image_info = images_info[ann['image_id']]
        image_file_name = image_info['file_name'].split('/')[1]
        path_to_image = os.path.join(image_dir, image_file_name)

        if not os.path.exists(path_to_image):
            print(image_file_name)
            continue


        if only_one_class:
            category_id = 0
        else:
            category_id = ann['category_id'] - 1

        width, height = image_info['width'], image_info['height']
        x_center = (ann['bbox'][0] + ann['bbox'][2] / 2) / width
        y_center = (ann['bbox'][1] + ann['bbox'][3] / 2) / height
        bbox_width = ann['bbox'][2] / width
        bbox_height = ann['bbox'][3] / height

        yolo_format = f"{category_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

        label_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        labels_output_path = os.path.join(output_dir, 'labels', label_file_name)
        images_output_path = os.path.join(output_dir, 'images', image_file_name)

        with open(labels_output_path, 'a') as file:
            file.write(yolo_format)

        shutil.copy(path_to_image, images_output_path)

root_dataset = '/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign_pothole'

label_map_file = f'{root_dataset}/label_map.json'
labels_path = f'{root_dataset}/labels.txt'

coco_json_train = f'{root_dataset}/train_anno.json'
coco_json_val = f'{root_dataset}/val_anno.json'

path_to_images = f'{root_dataset}/rtsd-frames/rtsd-frames'

path_yolo_dataset = '/datasets/traffic_sign_pothole/dataset'
output_train = '/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign_pothole/dataset/train'
output_valid = '/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign_pothole/dataset/valid'

samples_per_class = 100
min_area = 900
filter_annotation_id = None

only_one_class = False

convert_coco_to_yolo(coco_json_train,
                     output_train,
                     path_to_images,
                     filter_annotation_id,
                     only_one_class)
convert_coco_to_yolo(coco_json_val,
                     output_valid,
                     path_to_images,
                     only_one_class=only_one_class)
