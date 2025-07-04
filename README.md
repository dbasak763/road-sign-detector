# Road Sign Detection using YOLOv5

This project fine-tunes a YOLOv5 model on a custom dataset to detect four classes of road signs: `trafficlight`, `stop`, `speedlimit`, and `crosswalk`.

## Project Structure

- `yolov5/`: A submodule containing the official YOLOv5 repository.
- `Road_Sign_Dataset/`: Contains the images and labels for training, validation, and testing.
- `convert_annotations.py`: A script to convert PASCAL VOC XML annotations to the YOLOv5 `.txt` format.
- `partition_dataset.py`: A script to split the dataset into training, validation, and test sets.
- `test_annotations.py`: A utility script to visualize annotations on an image to verify correctness.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd road-sign-detector
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (like `conda` or `venv`).
    ```bash
    pip install -r yolov5/requirements.txt
    ```

## Data Preparation

This project uses a publicly available dataset of road signs originally annotated in PASCAL VOC format: [Road Sign Detection dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection).

1.  **Download and Unzip:**
    Download the dataset and place the `images` and `annotations` folders into a `Road_Sign_Dataset` directory in the project root.

2.  **Convert Annotations:**
    The original annotations are in PASCAL VOC XML format. Convert them to the required YOLOv5 `.txt` format. This will create a new `labels` directory.
    ```bash
    python convert_annotations.py
    ```

3.  **Partition Dataset:**
    Split the dataset into training (80%), validation (10%), and test (10%) sets. This script will organize the images and labels into `train`, `val`, and `test` subdirectories.
    ```bash
    python partition_dataset.py
    ```

## Training the Model

The model is trained on the CPU using the pre-trained `yolov5s.pt` weights. The dataset configuration is defined in `yolov5/data/road_sign_data.yaml`.

To start training, run the following command from within the `yolov5/` directory:

```bash
cd yolov5
python train.py --img 640 --batch 32 --epochs 100 --data road_sign_data.yaml --weights yolov5s.pt --name yolo_road_det --hyp data/hyps/hyp.scratch-low.yaml
```

The best trained model weights will be saved as `runs/train/yolo_road_det/weights/best.pt`.

## Evaluation

To evaluate the performance of the fine-tuned model on the test set, run the `val.py` script from the `yolov5/` directory:

```bash
python val.py --weights runs/train/yolo_road_det/weights/best.pt --data road_sign_data.yaml --task test --name yolo_det
```

The results, including mean Average Precision (mAP) and other metrics, will be saved in `runs/val/yolo_det/`.

## Inference

To run inference on new images and see the model in action, use the `detect.py` script from the `yolov5/` directory.

```bash
python detect.py --source ../Road_Sign_Dataset/images/test/ --weights runs/train/yolo_road_det/weights/best.pt --name yolo_road_det
```

The output images with bounding boxes drawn around the detected road signs will be saved in `runs/detect/yolo_road_det/`.
