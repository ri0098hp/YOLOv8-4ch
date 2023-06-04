# YOLOv8-4ch

## DEMO

<https://user-images.githubusercontent.com/104181368/222668654-0efb9323-4e6a-408c-90a1-a6807b9d8e40.mp4>

| Best Model    | Device            | Format        | Speed/Image | FPS  | AP@All |
| ------------- | ----------------- | ------------- | ----------- | ---- | ------ |
| YOLOv8s       | RTX3090           | PyTorch       | 2.7 ms      | 360  | 84.2   |
| (w/ augment)  |                   | TensorRT FP16 | 1.1 ms      | 909  | 84.0   |
|               | Colab (NVIDIA T4) | PyTorch       | 13.4 ms     | 74   | -      |
|               |                   | TensorRT FP16 | 3.0 ms      | 333  | -      |
|               | Jetson AGX  Orin  | PyTorch       | 19.9 ms     | 50   | -      |
|               |                   | TensorRT      | 1.7 ms      | 588  | -      |
|               | Intel i7-12700    | PyTorch       | 189.5 ms    | 5.2  | -      |
|               |                   | ONNX          | 47.5 ms     | 21.0 | -      |
| (w/o augment) | RTX3090           | PyTorch       | 2.8 ms      | 357  | 82.3   |
| YOLOv8-2L     |                   | PyTorch       | 2.0 ms      | 500  | 81.8   |
| (w/ augment)  |                   | TensorRT FP16 | 0.8 ms      | 1111 | 81.5   |
| YOLOv8-3L     |                   | PyTorch       | 2.6 ms      | 384  | 82.2   |
| (w/ augment)  |                   | TensorRT FP16 | - ms        | -    | -      |
| YOLOv8-4L     |                   | PyTorch       | 3.2 ms      | 312  | 82.3   |
| (w/ augment)  |                   | TensorRT FP16 | - ms        | -    | -      |

- 500 images are tested on Speed/Image, FPS
- fujinolab-all are tested on AP@All

## Original

 This is forked repo from (ultralytics/ultralytics) on commit a2bb42d.  
 Catch up to changes [here](https://github.com/ultralytics/ultralytics/compare/a2bb42d...main).

## Features

YOLOv8 をRGB-FIR向けに拡張したもの. 次の機能をオリジナルから追加している.

- [x] 1ch, 2ch, 3ch, 4chの学習・テストに対応 (ch=1等で指定可能)
- [x] 4chの推論に対応
- [x] TensorRT, ONNX, OpenVINOのエクスポートに対応
- [x] データセットの読み込みを確認、cfgの設定をできるツール `check_dataset.py`
- [x] テスト結果をsvg, csvで保存
- [x] テスト結果の画像を全て書き出し
- [x] pip installで利用可能にする

## TODO

基本的にはissueに投げる.  
開発をする際には自分のブランチを作成する. その後mainへマージする際には各ブランチ上でpull rebaseした後にPRを投げる.

```bash
git pull --rebase origin main
```

## 1. Installation

以下[1.1](#11-環境構築)以降の手法はYOLOの内部を弄る必要があるときである.  
ただ利用するだけならまず[Actions](https://github.com/Rits-Fujinolab/YOLOv8-4ch/actions/workflows/release.yaml)で`Run workflow`を実行すれば, [Release](https://github.com/Rits-Fujinolab/YOLOv8-4ch/releases/latest)に最新のwhlファイルが実装される.  
これをダウンロードしたのち

```bash
pip install [whlファイル]
```

でインストールをすればよい.

### 1.1 環境構築

必要に応じてDockerのセットアップやNVIDIA環境を入れる. [[参考]](<https://github.com/Rits-Fujinolab/Docker-setup/blob/master/server.md>)  

### 1.2 レポジトリを取得

レポジトリをcloneする. git環境がある人はOrganizationsにアクセス権限のあるuser名とmailを設定して

```bash
git clone git@github.com:Rits-Fujinolab/YOLOv3-4ch.git
```

または [GitHub CLI](https://cli.github.com) をインストールしてログイン認証後

```bash
gh repo clone Rits-Fujinolab/YOLOv3-4ch
```

### 1.3 コンテナを立ち上げる

devcontainerかdocker-compose upでコンテナを立ち上げる.

### 1.4 YOLOv8をpipでインストール

次のコマンドを実行してファイルの変更をパッケージに逐次反映出来るようにする.  

```bash
pip install -e ".[dev]"
```

### 1.5 データセットを準備

データセットをdatasetフォルダーに入れる.  
[dataloader](utils/datasets.py) を魔改造してるため次のようなディレクトリ構造推奨...  
シンボリックリンクでも認識可能なのでデータフォルダを作った後, フォルダごとにリンクを作るとスペースを節約できる.

```txt
  <datasets>
  ├── fujinolab-all
  │   ├── 20180903_1113
  │   │   ├── FIR
  │   │   ├── FIR_labels
  │   │   ├── RGB
  │   │   ├── RGB_crop
  │   │   └── RGB_raw
  │   └── 20190116_2008
  │       ├── FIR
  │       ├── FIR_labels
  │       ├── RGB
  │       ├── RGB_crop
  │       ├── RGB_labels
  │       └── RGB_raw
  ├── fujinolab-day
  │   └── 20180903_1113  <-シンボリックリンク推奨
  │   
  └── kaist-all
      ├── train
      │   ├── set00
      │   │   ├──V000
      │   │   │  ├── labels
      │   │   │  ├── person
      │   │   │  ├── lwir
      │   │   │  └── visible
      │   │   └──V001
      │   │      ├── labels
      │   │      ├── person
      │   │      ├── lwir
      │   │      └── visible
      │   └── set01
      └── val
```

## 2. Usage (CLI)

### 2.1 データセットの準備

[ここ](data/fujinolab-all.yaml)を参考にディレクトリとクラスを指定.  
なおRGB画像、FIR画像、ラベルファイルは全て同じ名前を持っている必要がある (RGBを基準に出ディレクトリを置換している)

### 2.2 訓練

[ここ](memo.txt)を参照.  
基本的にはdataオプションとbatch-sizeオプション, epochsオプションで変更すればよい.  
以下例...

```bash
yolo detect train data=[yamlへのパス] model=[yamlまたはptへのパス]
```

### 2.3 テスト

[ここ](memo.txt)を参照. 基本的にはdataオプションと重みファイルオプションで変更すればよい.  
スライド用の画像を探す場合はsave-allオプションが便利.  
以下例...

```bash
yolo detect val model=[重みファイルへのパス] data=[データyamlへのパス] save-all
```

### 2.4 検知

```bash
yolo detect predict model=[重みファイルへのパス] source=[データフォルダ] [検知結果画像を出力する場合は"save"]
```

### 2.5 モデル変換

NVIDIAデバイス向けのTensorRTやIntelデバイス向けのOpenVINO, オープンソース(主にAMDデバイス向け)のONNXへの変換を行うことができる.

```bash
yolo export model=[重みファイルへのパス] format=[フォーマット]
```

なおformatは[engine, openvino, onnx, tflite]など [[詳細](https://docs.ultralytics.com/modes/export/)].

### 2.6 一連のテスト

[ここ](memo.txt)を参照. 基本的にはdataオプションとbatch-sizeオプション, epochsオプションで変更すればよい.  
場合によってはdataやモデル名を指定して変更すること. test時には2000枚ほどを使用して訓練が行われる.
以下例...

```bash
yolo cfg=cfg/test.yaml
```

## 3. Usage (Python with pip)

### 3.1 データセットの準備

[ここ](data/fujinolab-all.yaml)を参考にディレクトリとクラスを指定.  
なおRGB画像、FIR画像、ラベルファイルは全て同じ名前を持っている必要がある (RGBを基準に出ディレクトリを置換している)

### 3.2 Pythonコードで利用

YOLOv8 may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch
# OR
model = YOLO("weights/fujinolab-all-4ch-aug.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="data/fujinolab-all.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("datasets/demo")  # predict on an image
success = model.export(format="onnx")  # export the model to ONNX format
```

## 4. Jetson メモ

YOLOv5のドキュメントを参照. [[Web](https://docs.ultralytics.com/yolov5/tutorials/running_on_jetson_nano/)] 
またコミュニティも参照のこと. [[参考](https://elinux.org/Jetson_Zoo)]

### 4.1 CUDA Pythonのビルド

非常に時間がかかる.

```bash
wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-7-0.sh
sudo chmod 755 ./OpenCV-4-7-0.sh
```

ダウンロードしたシェルスクリプトの17行目`sudo apt-get install -y python-dev python-numpy python-pip`を削除する.  
その後実行 (30分くらい).  

```bash
./OpenCV-4-7-0.sh
```

### 4.2 環境構築

Dockerを使っても良いがpipenvやvenvによる仮想化の方が処理速度的に便利.  
`requirements.txt`を編集して`torch`, `torchvision`, `onnx-gpu`などGPU関連のパッケージをコメントアウトする.  
その後環境を構築する.  
なおtorchとtorchvisionはその時々によって変わるので公式HP参照.  
[[JetPackとtorchのバージョン関係](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#pytorch-jetson-rel)]
[[torchのインストールURL一覧](https://developer.download.nvidia.cn/compute/redist/jp/)]
[[torchとtorchvisionのバージョン関係](https://github.com/pytorch/vision#installation)]  

1. 仮想環境の構築

```bash
pipenv --python /usr/bin/python3 --site-packages install
```

2. torchのインストール

```bash
pipenv install https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0a0+fe05266f.nv23.04-cp38-cp38-linux_aarch64.whl
```

3. torchvisionのインストール (時間がかかる)

```bash
git clone --branch v0.15.2 https://github.com/pytorch/vision torchvision
cd torchvision
pipenv run python3 setup.py install
cd .. && rm -rf torchvision
```

4. YOLOv8のインストール

```bash
pipenv install -e ".[dev]"
```

5. 必要に応じて次もインストール

ONNX系 [[参考](https://elinux.org/Jetson_Zoo#ONNX_Runtime)]

```bash
wget https://nvidia.box.com/shared/static/amhb62mzes4flhbfavoa73m5z933pv75.whl -O onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
pipenv install onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
rm onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
```
