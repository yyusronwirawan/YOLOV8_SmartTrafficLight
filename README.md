## Road-detection: detection of traffic signs and potholes by ML-models
Implementation of detection by transfer learning with popular Yolo8 and Fast-RCNN from Detectrons zoo. Training takes place on a custom dataset assembled from several.

![screenshot](https://github.com/alinzh/road_detection/blob/main/data/processed_img/traffic_sign_pothole/example.jpg)

The dataset includes 157 classes:
- 156 - different types of Russian traffic signs
- 1 - potholes

The repository contains a script for assembling datasets and bringing them to required annotation format.

Both datasets are available at the following links:
- [potholes](https://www.dropbox.com/s/qvglw8pqo16769f/pothole_dataset_v8.zip?dl=1)
- [Russian traffic signs](https://www.kaggle.com/datasets/watchman/rtsd-dataset/data)
 
Download weights for trained Yolo8m from [here](https://drive.google.com/file/d/1H4E8yIXL7RlM6tRJayRhyHlWpOJd0L5Y/view?usp=sharing).
## Summary
Not bad!
![summary](https://github.com/alinzh/road_detection/blob/main/summary/summary_yolo8.png)                  
