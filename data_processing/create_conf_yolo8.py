import clearml
import yaml


path_yolo_dataset = '/datasets/traffic_sign_pothole/dataset'
root_dataset = '/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign_pothole'
only_one_class = False
labels_path = f'{root_dataset}/labels.txt'

data = {
    'path': path_yolo_dataset,
    'train': 'train/images',
    'val': 'valid/images',
}

if only_one_class:
    data['names'] = {
        0: 'traffic_sign_pothole'
    }
else:
    with open(labels_path, 'r') as file:
        class_names = [line.strip() for line in file]
        id2class = dict(zip(range(len(class_names)), class_names))
        data['names'] = id2class

with open('/datasets/traffic_sign_pothole/russian_trafic_signs.yaml', 'w') as file:
    yaml.dump(data, file)

clearml.browser_login()
