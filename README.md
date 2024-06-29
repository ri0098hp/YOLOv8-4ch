# YOLOv8-4ch

[English](README.md) | [日本語🇯🇵](README-ja.md)

## DEMO

<https://github.com/ri0098hp/YOLOv8-4ch/assets/104181368/177f036e-4932-4fda-9338-532504e81663>

| **Model**       | **Device**       | **Format**    | **Speed/Image** | **FPS** | **AP@All** | **LAMR50** |
| --------------- | ---------------- | ------------- | --------------- | ------- | ---------- | ---------- |
| YOLOv8s-2stream | RTX3090          | PyTorch       | 4.1 ms          | 241     | 86.2       | 18.5       |
|                 |                  | TensorRT FP16 | 1.5 ms          | 675     | 85.9       | 18.7       |
|                 | Jetson AGX  Orin | PyTorch       | 16.2 ms         | 61      | 86.5       | 17.9       |
|                 |                  | TensorRT FP16 | 13.7 ms         | 72      | 86.0       | 18.7       |
|                 | Intel i7-12700   | ONNX          | 73.0 ms         | 13      | 85.9       | 18.7       |

- AP means AP@0.5 in a single class.
- Tested on All-Season-Dataset (Ours)
- ultralytics 8.1.29
- Jetson uses JetPack 6.0 DP

## Paper Results

### 4-CHANNEL

| Model | AP@Hot   | AP@Inter | AP@Cold  | Avg.     |
| ----- | -------- | -------- | -------- | -------- |
| Hot   | **74.5** | 81.8     | 69.5     | 75.3     |
| Inter | 50.2     | 84.7     | 68.0     | 67.6     |
| Cold  | 49.6     | 82.3     | **77.7** | 69.9     |
| Mix   | 67.5     | **85.0** | 75.6     | **76.0** |

### 2-STREAM

| Model |  AP@Hot  | AP@Inter | AP@Cold  |   Avg.   |
| :---- | :------: | :------: | :------: | :------: |
| Hot   | **78.3** |   84.2   |   72.0   |   78.2   |
| Inter |   56.3   |   85.9   |   72.2   |   71.5   |
| Cold  |   59.8   |   85.2   |   80.1   |   75.0   |
| Mix   |   77.5   | **87.8** | **80.9** | **82.1** |

## Original

 This is forked repo from (ultralytics/ultralytics) on commit 6909f08 (ultralytics 8.2.13).  
 Catch up to changes [here](https://github.com/ultralytics/ultralytics/compare/6909f08...main).

## Features

An extension of YOLOv8 for RGB-FIR. The following features have been added from the original:

- [x] Support for 1ch, 3ch, 4ch data (specified as ch=1 etc.)
- [x] Support for LAMR metric
- [x] Tool for checking dataset loading and configuring
- [x] Save test results in svg, csv
- [x] Export all test images
- [x] Available with pip install for 4ch dependencies
- [x] Implementation on Jetson
- [x] YOLOv8-2stream, a two-input model based on YOLOv8
- [x] Creating a saliency map using CAM

## Models

Baseline model: YOLOv8s-2stream [details](ultralytics/cfg/models)

![yolov8s-2stream.drawio.svg](ultralytics/cfg/models/diagram/yolov8s-2stream.drawio.svg)

## 1. Installation

If you just want to use it, the latest wheel file is available in [Release](https://github.com/ri0098hp/YOLOv8-4ch/releases/latest).  
After downloading this, install the package as below:

```bash
python -m venv .venv
. ./.venv/bin/activate
pip install ultralytics-YYYY.MM.DD-py3-none-any.whl[4ch]
```

Please follow the instructions to use it on the [CLI](#4-usage-cli) or Python code like in [scripts](scripts).  
We also prepare the demo code with datasets.  
Please refer to the next section.

## 2 Custom Installation and Demo Run

### 2.1 Full Installation

This is an installation way to customize our codes.
Set up Docker, NVIDIA Driver, and git environment as necessary.  
(We recommend use docker and devcontainer with VSCode.)  
Clone the repository.

```bash
git clone git@github.com:ri0098hp/YOLOv8-4ch.git
```

Or you can use [GitHub CLI](https://cli.github.com) with:

```bash
gh repo clone ri0098hp/YOLOv8-4ch
```

After that, start the container with devcontainer or docker-compose up.  
Or you can use virtualenv like venv like:

```bash
cd YOLOv8-4ch
python -m venv .venv
. ./.venv/bin/activate
```

After that, execute the following command in the directory where `pyproject.toml` is located.  
This allows you to reflect changes to the file sequentially in the package.

```bash
pip install -e .[4ch]
```

### 2.2 Run Demo Tests

Demo codes are available on [sweep_val_season.py](scripts/sweep_val_season.py).  
Please download the codes, weight files, and dataset from release v2024.06.13 assets [[here]](https://github.com/ri0098hp/YOLOv8-4ch/releases/tag/v2024.06.13).  
Note that the locations of each subset in the datasets ([hot](ultralytics/cfg/datasets/All-Season-hot.yaml), [inter](ultralytics/cfg/datasets/All-Season-inter.yaml), [cold](ultralytics/cfg/datasets/All-Season-cold.yaml)) in yaml file should be changed.  
(We recommend make `datasets` folder and place `All-Season-tiny` folder in it.)  
You must change the path to weight files, such as L8 and L25, in the [script](scripts/sweep_val_season.py).  
(We recommend make `runs` folder and place pt files in it.)  
Then, run the python command and you will obtain the result table as csv file.

```bash
python scripts/sweep_val_season.py
```

## 3. Dataset

Put the dataset in the dataset folder.  
Because the dataloader has been modified, the following directory structure is recommended.  
Except for All-Season, the search will be performed recursively under the train and val folders.  
Note that the labels, RGB images, and FIR images must exist in a symmetrical path relationship.  
Symbolic links can also be recognized, so after creating the data folder, you can save space by creating links for each folder.

```bash
mkdir train
cd train
find ../../All-Season/train -mindepth 1 -maxdepth 1 -type d -exec ln -s {} \;
cd ..
mkdir val
cd val
find ../../All-Season/val -mindepth 1 -maxdepth 1 -type d -exec ln -s {} \;
```

After that, specify the directory and class by referring to [`All-Season.yaml`](data/All-Season.yaml).  
The RGB image, FIR image, and label file must all have the same name. (The output directory is replaced based on RGB.)  
You can check whether it was successfully loaded by using the configuration or data YAML file with the following command.

```bash
yolo utils dataset
```

An example of the directory structure:

```txt
  <datasets>
  ├── All-Season (not available now)
  │   ├── train
  │   │   ├── 20180731_1415
  │   │   │   ├── set00
  │   │   │   │   ├── FIR
  │   │   │   │   ├── labels
  │   │   │   │   └── RGB
  │   │   │   └── set01
  │   │   └── 20190116_2008
  │   └── val
  ├── All-Season-hot (not available now)
  │   ├── train
  │   │   └── 20180731_1415 <-recommended to use symbolic links
  │   └── val
  ├── All-Season-tiny
  │   ├── hot
  │   ├── inter
  │   └── cold
  └── kaist-sanit (other datasets)
      ├── train
      │   ├── set00
      │   │   ├──V000
      │   │   │  ├── labels
      │   │   │  ├── lwir
      │   │   │  └── visible
      │   │   └──V001
      │   └── set01
      └── val
```

## 4. Usage (CLI)

### 4.1 Preparing the CFG File

Tweak the settings based on [`default.yaml`](ultralytics/cfg/default.yaml).  
For parameters such as `pos_imgs_train`, it is a good idea to use [`check_dataset.py`](ultralytics/utils/check_dataset.py).

```bash
yolo utils dataset
```

Also, all parameters can be changed through commands, so there is no need to force separate cfg files.  
Additional parameters are available like below:

|   Parameters    | Type  | Description                                                                    |
| :-------------: | :---: | :----------------------------------------------------------------------------- |
|       ch        |  int  | Number of channels in the data set                                             |
|    save_all     | bool  | Save all result images during testing                                          |
| pos_imgs_train  |  int  | Number of labeled images during training (≠ number of instances)               |
| neg_ratio_train | float | Ratio of unlabeled images to all images during training                        |
|     hsv_ir      | float | Data augmentation. Amplitude ratio for changing brightness value of FIR image. |
|     flipir      | float | Data augmentation. Establishment of black and white inversion of FIR image.    |

### 4.2 Train, Validate, and Predict

All commands for basic method is the same as official implementation.  
Therefore, please see and follow their documents. [[here]](https://docs.ultralytics.com)

### 4.3 CAM Visualization

Visualize the attention of the model by GradCAM or LaryerCAM on the image with a heat map.
The command is as follows, directly under the workspace.

```bash
yolo utils gradcam source=[data folder or image file path] model=[path to weight file] layer=[layer selection]
```

The supported arguments are as follows, including the above example.

| Parameter name |      type       |           Default            | Description                                                                                                                      |
| :------------: | :-------------: | :--------------------------: | :------------------------------------------------------------------------------------------------------------------------------- |
|     source     |       str       | `ultralytics/assets/bus.jpg` | Directory path including RGB and FIR folders, or file path of JPEG image                                                         |
|    project     |       str       |        `runs/gradcam`        | Root path of destination                                                                                                         |
|      name      |       str       |       Image file name        | Destination folder name                                                                                                          |
| backward_type  | [class,box,all] |            class             | Type of output to be backpropagated. Confidence, bbox coordinates, both                                                          |
|   conf, iou    |      float      |           0.1,0.25           | Confidence and IoU threshold                                                                                                     |
|     model      |       pt        |          yolov8n.pt          | Path of weight file (official weights are automatically downloaded)                                                              |
|     method     |    See below    |           XGradCAM           | CAM type                                                                                                                         |
|     layer      |  list[int,...]  |          [15,18,21]          | Location of the layer that uses the feature map. Recommended focusing on just before the Detect layer. (For 2stream, [29,32,35]) |
|  renormalize   |      bool       |            False             | Normalize the heat map within the bbox. Mainly used for class classification considerations.                                     |
|    show_box    |      bool       |             True             | Display the bbox of the detected object                                                                                          |
|      only      |  ["",RGB,FIR]   |              ""              | Perform detection using only the data specified by the RGB-FIR detector.                                                         |

- Supported CAM types
  - Gradient required: GradCAM, GradCAMPlusPlus, EigenGradCAM, LayerCAM, HiResCAM, XGradCAM
  - Gradient free: EigenCAM, RandomCAM(?)

## 5. Jetson Implementation

This document explains the procedure based on JetPack 6.0 DP.  
Please read the official documents on [GitHub](https://github.com/ultralytics/ultralytics/blob/5f7d76e2eb50d50873825bcd3e675537b2396dd3/docs/en/guides/nvidia-jetson.md)

### 5.1 Building and Installing the Library

Install `torch`, `torchvision`, and `cv2` to Python on Tegra OS.  
If they are already installed, go to the next section (#42-Installing yolo).  
Use the script [`build_jetson.sh`] (scripts/build_jetson.sh) created below.  
Note that JetPack, torch, and torchvision change from time to time, so refer to the following URL to rewrite the script.  
[[JetPack and torch version relationship](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#pytorch-jetson-rel)]  
[[torch installation URL list](https://developer.download.nvidia.com/compute/redist/jp/)]  
[[torch and torchvision version relationship](https://github.com/pytorch/vision#installation)]  
[[JetPack and onnx_runtime version relationship](https://elinux.org/Jetson_Zoo#ONNX_Runtime)]

### 5.2 Installing YOLOv8-4ch

First, use venv to use libraries included in the Python on your system.
Libraries on system, such as `torch`, `torchvision`, `onnxrutime_gpu`, `cv2`, etc. are should be import.

```bash
python3 -m venv venv --system-site-packages
```

Edit [`pyproject.toml`](pyproject.toml) and comment out GPU-related packages such as `torch`, `torchvision`, and `onnx-gpu`.
Then pip install YOLO in development mode `-e`.

```bash
source venv/bin/activate
pip install -e .[4ch]
```

## 6. Citation

```bibtex
@INPROCEEDINGS{10325231,
  author={Okuda, Masato and Yoshida, Kota and Fujino, Takeshi},
  booktitle={2023 IEEE SENSORS}, 
  title={Multispectral Pedestrian Detection with Visible and Far-infrared Images Under Drifting Ambient Light and Temperature}, 
  year={2023},
  volume={},
  number={},
  pages={1-4},
  doi={10.1109/SENSORS56945.2023.10325231}
}
```
