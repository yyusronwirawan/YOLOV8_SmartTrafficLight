import os

path_pothole_da_dir = "/Users/alina/PycharmProjects/obj_detection/datasets/pothole_dataset_v8"
test_labels_dir = f"{path_pothole_da_dir}/valid/labels/"
train_labels_dir = f"{path_pothole_da_dir}/train/labels/"
test_img_dir = f"{path_pothole_da_dir}/valid/images/"
train_img_dir = f"{path_pothole_da_dir}/train/images/"

test_labels_list = os.listdir(test_labels_dir)
train_labels_list = os.listdir(train_labels_dir)
test_img_list = os.listdir(test_labels_dir)
train_img_list = os.listdir(train_labels_dir)


def change_class_idx():
    # change class in test data
    for path in test_labels_list:
        with open(f"{test_labels_dir}{path}", "r") as file:
            new_data = ''
            for line in file:
                l_new = line[1:]
                l_new = '155' + l_new
                new_data += l_new

        with open(f"{test_labels_dir}{path}", "w") as file:
            file.write(new_data)

    # change class in train data
    for path in train_labels_list:
        with open(f"{train_labels_dir}{path}", "r") as file:
            new_data = ''
            for line in file:
                l_new = line[1:]
                l_new = '155' + l_new
                new_data += l_new

        with open(f"{train_labels_dir}{path}", "w") as file:
            file.write(new_data)


if __name__ == "__main__":
    change_class_idx()









