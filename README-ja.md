# YOLOv8-4ch

[English 🇺🇸](README.md) | [日本語 🇯🇵](README-ja.md) | [Paper](https://doi.org/10.1109/OJITS.2024.3507917)

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
- Batch-size=1 on Speed/Image, FPS
- Jetson uses JetPack 6.0 DP

## Original

 This is forked repo from (ultralytics/ultralytics) on commit 6909f08 (ultralytics 8.2.13).  
 Catch up to changes [here](https://github.com/ultralytics/ultralytics/compare/6909f08...main).

## Features

YOLOv8 をRGB-FIR向けに拡張したもの. 次の機能をオリジナルから追加している.

- [x] 1ch, 3ch, 4chの学習・テスト・推論に対応 (ch=1等で指定可能)
- [x] TensorRT, ONNX, OpenVINOのエクスポートに対応
- [x] データセットの読み込みを確認、cfgの設定をできるツール `check_dataset.py`
- [x] テスト結果をsvg, csvで保存
- [x] テスト結果の画像を全て書き出し
- [x] pip installで利用可能にする
- [x] Jetsonで実装
- [x] YOLOv8をベースにした2入力モデルYOLOv8-2stream
- [x] CAMによるサリエンシーマップの作成

## Models

ベースラインとなっているモデル. [詳細](ultralytics/cfg/models)

![yolov8s-2stream.drawio.svg](ultralytics/cfg/models/diagram/yolov8s-2stream.drawio.svg)

## 1. Installation

以下[1.1](#11-環境構築)以降の手法はYOLOの内部を弄る必要があるときである.  
ただ利用するだけなら, まず[Actions](https://github.com/ri0098hp/YOLOv8-4ch/actions/workflows/release.yaml)で`Run workflow`を実行すれば, [Release](https://github.com/ri0098hp/YOLOv8-4ch/releases/latest)に最新のwhlファイルが実装される.  
これをダウンロードしたのち

```bash
pip install [whlファイル]
```

でインストールした後, [CLI](#2-usage-cli)上や[Pythonコード](#3-usage-python-with-pip)上で従って利用する.

### 1.1 環境構築

必要に応じてDockerのセットアップやNVIDIA環境を入れる. [[参考]](<https://github.com/Rits-Fujinolab/Docker-setup/blob/master/server.md>)  

### 1.2 レポジトリを取得

レポジトリをcloneする. git環境がある人はOrganizationsにアクセス権限のあるuser名とmailを設定して

```bash
git clone git@github.com:ri0098hp/YOLOv8-4ch.git
```

または [GitHub CLI](https://cli.github.com) をインストールしてログイン認証後

```bash
gh repo clone ri0098hp/YOLOv8-4ch
```

### 1.3 コンテナを立ち上げる

devcontainerかdocker-compose upでコンテナを立ち上げる.

### 1.4 YOLOv8をpipでインストール

`pyproject.toml`があるディレクトリで次のコマンドを実行する.  
これにより, 逐次ファイルの変更をパッケージに反映出来るようにする.  
なおオプションとして最後の`.`をONNXやTensorRTに対応する`.[export]`に変えることも可能.

```bash
pip install -e .[4ch]
```

### 1.5 データセットを準備

データセットをdatasetフォルダーに入れる.  
dataloaderを魔改造してるため次のようなディレクトリ構造推奨...  
All-Season以外はtrainとvalフォルダ以下で再帰的に探索を行う.  
尚ラベルとRGB画像とFIR画像は対称となるパス関係に存在する必要がある.  
シンボリックリンクでも認識可能なのでデータフォルダを作った後, フォルダごとにリンクを作るとスペースを節約できる

```bash
mkdir train
cd train
find ../../All-Season/train -mindepth 1 -maxdepth 1 -type d -exec ln -s {} \;
cd ..
mkdir val
cd val
find ../../All-Season/val -mindepth 1 -maxdepth 1 -type d -exec ln -s {} \;
```

その後, [`All-Season.yaml`](data/All-Season.yaml) を参考にディレクトリとクラスを指定.  
なおRGB画像、FIR画像、ラベルファイルは全て同じ名前を持っている必要がある. (RGBを基準に出ディレクトリを置換している)  
上手く読み込めるかどうかは次のコマンドでcfgやdataのYAMLファイルを利用して確認できる.

```bash
yolo utils dataset
```

以下はディレクトリ構造の例.

```txt
  <datasets>
  ├── All-Season
  │   ├── train
  │   │   ├── 20180731_1415
  │   │   │   ├── set00
  │   │   │   │   ├── FIR
  │   │   │   │   ├── labels
  │   │   │   │   └── RGB
  │   │   │   └── set01
  │   │   └── 20190116_2008
  │   └── val
  ├── All-Season-hot
  │   ├── train
  │   │   └── 20180731_1415 <-シンボリックリンク推奨
  │   └── val
  └── kaist-sanit
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

## 2. Usage (CLI)

### 2.1 cfgファイルの準備

[`default.yaml`](ultralytics/cfg/default.yaml) をベースに設定を弄る.  
なお `pos_imgs_train` などのパラメータは [`check_dataset.py`](ultralytics/utils/check_dataset.py) を活用すると良い.  

```bash
yolo utils dataset
```

またパラメータは全てコマンド上でも変更可能であるから, 無理にcfgファイルを分ける必要はない.  
以下は追加パラメータの説明.

|  パラメータ名   | type  | 説明                                             |
| :-------------: | :---: | :----------------------------------------------- |
|       ch        |  int  | データセットのチャネル数                         |
|    save_all     | bool  | テスト時に全ての結果画像を保存する               |
| pos_imgs_train  |  int  | 学習時のラベル有り画像の枚数 ( ≠ インスタンス数) |
|  pos_imgs_val   |       | 訓練時の同上                                     |
| neg_ratio_train | float | 学習時のラベル無し画像の全画像に対する比率       |
|  neg_ratio_val  |       | 訓練時の同上                                     |
|     hsv_ir      | float | データ拡張. FIR画像の輝度値を変化させる振幅割合. |
|     flipir      | float | データ拡張. FIR画像の白黒反転の確立.             |

### 2.2 訓練

[ここ](memo.txt)を参照.  
必要に応じて`cfg`オプションを読み込めばよい.  
基本的には`data`オプションと`batch`オプション, `epochs`オプションで変更すればよい.

```bash
yolo detect train data=[yamlへのパス] model=[yamlまたはptへのパス]
```

### 2.3 テスト

基本的には`data`オプションと重みファイルオプションで変更すればよい.  
速度を測定する場合には`batch=1`とする.  
スライド用の画像を探す場合は`save_all`オプションが便利.

```bash
yolo detect val model=[重みファイルへのパス] data=[データyamlへのパス] save_all
```

### 2.4 検知

`source`で指定するフォルダ直下にRGB, FIRのいずれか, あるいは両方が存在する必要がある.

```bash
yolo detect predict model=[重みファイルへのパス] source=[データフォルダ] save
```

### 2.5 モデル変換

NVIDIAデバイス向けのTensorRTやIntelデバイス向けのOpenVINO, オープンソース(主にAMDデバイス向け)のONNXへの変換を行うことができる.  
必要に応じて半精度`half`オプションを入れる.  
なおformatは[engine, openvino, onnx, tflite]など [[詳細](https://docs.ultralytics.com/modes/export/)].

```bash
yolo export model=[重みファイルへのパス] format=[フォーマット] half
```

### 2.6 CAMによる注目領域の可視化

GradCAMやLaryerCAMによるモデルの注目度を画像上にヒートマップで可視化する.  
コマンドはワークスペース直下にて以下の通り.

```bash
yolo utils gradcam source=[データフォルダ or 画像ファイルパス] model=[重みファイルへのパス] layer=[レイヤの選択]
```

対応している引数は上記の例も含めて以下の通り.

| パラメータ名  |      type       |          デフォルト          | 説明                                                                              |
| :-----------: | :-------------: | :--------------------------: | :-------------------------------------------------------------------------------- |
|    source     |       str       | `ultralytics/assets/bus.jpg` | RGB・FIRフォルダを含むディレクトリパス, またはJPEG画像のファイルパス              |
|    project    |       str       |        `runs/gradcam`        | 保存先のルートパス                                                                |
|     name      |       str       |        画像ファイル名        | 保存先のフォルダ名                                                                |
| backward_type | [class,box,all] |            class             | 逆伝搬させる出力の種類. 信頼度, bbox座標, 両方                                    |
|   conf, iou   |      float      |           0.1,0.25           | 信頼度とIoUの閾値                                                                 |
|     model     |       pt        |          yolov8n.pt          | 重みファイルのパス (公式の重みは自動ダウンロード)                                 |
|    method     |    下記参照     |           XGradCAM           | CAMの種類                                                                         |
|     layer     |  list[int,...]  |          [15,18,21]          | 特徴量マップを利用するレイヤの場所. Detect層の直前を推奨. (2streamなら[29,32,35]) |
|  renormalize  |      bool       |            False             | bbox内でヒートマップを正規化する. 主にクラス分類に対する考察で使用する.           |
|   show_box    |      bool       |             True             | 検出したオブジェクトのbboxを表示する                                              |
|     only      |  ["",RGB,FIR]   |              ""              | RGB-FIR検出器で指定した方のみのデータで検出を行う.                                |

- 対応しているCAMの種類
  - Gradient required: GradCAM, GradCAMPlusPlus, EigenGradCAM, LayerCAM, HiResCAM, XGradCAM
  - Gradient free: EigenCAM, RandomCAM(?)

### 2.7 動作テスト

基本的にはdataオプションとbatch-sizeオプション, epochsオプションで変更すればよい.  
場合によってはdataやモデル名を指定して変更すること. test時には2000枚ほどを使用して訓練が行われる.

```bash
yolo cfg=test.yaml
```

## 3. Usage (Python with pip)

CLI以外にPythonやJupyter上でも使用できる.  
使用方法は [公式ドキュメント](https://docs.ultralytics.com/) 参照.  
なお組み込み例は [こちら](https://github.com/ri0098hp/harvesters4RGB-FIR) を参照.

## 4. Jetson メモ

このドキュメントではJetPack 6.0 DPを基準に説明する.

### 4.1 ライブラリのビルドとインストール

Tegra OS上のPythonに`torch`や`torchvision`, `cv2`をインストールする.  
既にインストールされている場合は[次の節](#42-yoloのインストール)へ.  
以下に作成したスクリプト[`build_jetson.sh`](scripts/build_jetson.sh)を利用する.  
なおJetPackとtorchやtorchvisionはその時々によって変わるの以下のURLを参照しスクリプトを書き換える.  
[[JetPackとtorchのバージョン関係](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#pytorch-jetson-rel)], [[torchのインストールURL一覧](https://developer.download.nvidia.com/compute/redist/jp/)], [[torchとtorchvisionのバージョン関係](https://github.com/pytorch/vision#installation)], [[JetPackとonnx_runtimeのバージョン関係](https://elinux.org/Jetson_Zoo#ONNX_Runtime)]

### 4.2 YOLOのインストール

#### 4.2.1 仮想環境の構築

まずvenvを使用してシステム上のPythonに含まれるライブラリを使用する.  
`torch`や`torchvision`, `onnxrutime_gpu`や`cv2`などを引き継ぐ.  
(pipenvやpoetryはハッシュ計算や構造が面倒で, 宗教上の理由がない人は非推奨)

```bash
python3 -m venv venv --system-site-packages
```

#### 4.2.2 YOLOv8のインストール

[`pyproject.toml`](pyproject.toml)を編集して`torch`, `torchvision`, `onnx-gpu`などGPU関連のパッケージをコメントアウトする.  
その後YOLOを開発モード`-e`でpipインストールする.

```bash
source venv/bin/activate
pip install -e .
```

## 5. Citation

```bibtex
@article{10770282,
  author   = {Okuda, Masato and Yoshida, Kota and Fujino, Takeshi},
  journal  = {IEEE Open Journal of Intelligent Transportation Systems},
  title    = {Realtime Multispectral Pedestrian Detection With Visible and Far-Infrared Under Ambient Temperature Changing},
  year     = {2024},
  volume   = {5},
  number   = {},
  pages    = {797-809},
  keywords = {Finite impulse response filters;Cameras;Pedestrians;Accuracy;Image edge detection;Feature extraction;Deep learning;YOLO;Training;Synchronization;Object detection;pedestrian detection;deep learning;far-infrared;sensor fusion},
  doi      = {10.1109/OJITS.2024.3507917}
}
