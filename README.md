# YOLOv8-4ch

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

 This is forked repo from (ultralytics/ultralytics) on commit a62cdab (ultralytics 8.1.29).  
 Catch up to changes [here](https://github.com/ultralytics/ultralytics/compare/a62cdab...main).

## Features

YOLOv8 をRGB-FIR向けに拡張したもの. 次の機能をオリジナルから追加している.

- [x] 1ch, 3ch, 4chの学習・テスト・推論に対応 (ch=1等で指定可能)
- [x] TensorRT, ONNX, OpenVINOのエクスポートに対応
- [x] データセットの読み込みを確認、cfgの設定をできるツール `check_dataset.py`
- [x] テスト結果をsvg, csvで保存
- [x] テスト結果の画像を全て書き出し
- [x] pip installで利用可能にする
- [x] Jetsonで実装
- [x] YOLOv8をベースにした2入力モデル

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
pip install -e .
```

### 1.5 データセットを準備

データセットをdatasetフォルダーに入れる.  
dataloaderを魔改造してるため次のようなディレクトリ構造推奨...  
All-Season以外はtrainとvalフォルダ以下で再帰的に探索を行う.  
尚ラベルとRGB画像とFIR画像は対称となるパス関係に存在する必要がある.  
シンボリックリンクでも認識可能なのでデータフォルダを作った後, フォルダごとにリンクを作るとスペースを節約できる

```bash
find ../All-Season -mindepth 1 -maxdepth 1 -type d -exec ln -s {} \;
```

その後, [`All-Season.yaml`](data/All-Season.yaml) を参考にディレクトリとクラスを指定.  
なおRGB画像、FIR画像、ラベルファイルは全て同じ名前を持っている必要がある. (RGBを基準に出ディレクトリを置換している)  
上手く読み込めるかどうかは [`check_dataset.py`](scripts/check_dataset.py) で確認できる.  
以下はディレクトリ構造の例.

```txt
  <datasets>
  ├── All-Season
  │   ├── 20180903_1113
  │   │   ├── FIR
  │   │   ├── FIR_labels
  │   │   └── RGB_homo
  │   └── 20190116_2008
  ├── All-Season-day
  │   └── 20180903_1113  <-シンボリックリンク推奨
  │   
  └── kaist-sanit
      ├── train
      │   ├── set00
      │   │   ├──V000
      │   │   │  ├── labels
      │   │   │  ├── person
      │   │   │  ├── lwir
      │   │   │  └── visible
      │   │   └──V001
      │   └── set01
      └── val
```

## 2. Usage (CLI)

### 2.1 cfgファイルの準備

[`default.yaml`](ultralytics/cfg/default.yaml) をベースに設定を弄る.  
なお `pos_imgs_train` などのパラメータは [`check_dataset.py`](scripts/check_dataset.py) を活用すると良い.  
またパラメータは全てコマンド上でも変更可能であるから, 無理にcfgファイルを分ける必要はない.  
以下は追加パラメータの説明.

| パラメータ名    | type  | 説明                                             |
| --------------- | ----- | ------------------------------------------------ |
| ch              | int   | データセットのチャネル数                         |
| save_all        | bool  | テスト時に全ての結果画像を保存する               |
| pos_imgs_train  | int   | 学習時のラベル有り画像の枚数 ( ≠ インスタンス数) |
| pos_imgs_val    |       | 訓練時の同上                                     |
| neg_ratio_train | float | 学習時のラベル無し画像の全画像に対する比率       |
| neg_ratio_val   |       | 訓練時の同上                                     |
| hsv_ir          | float | データ拡張. FIR画像の輝度値を変化させる振幅割合. |
| flipir          | float | データ拡張. FIR画像の白黒反転の確立.             |

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

### 2.6 動作テスト

[ここ](memo.txt)を参照. 基本的にはdataオプションとbatch-sizeオプション, epochsオプションで変更すればよい.  
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
既にインストールされている場合は[次の節](#422-yoloのインストール)へ.  
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
