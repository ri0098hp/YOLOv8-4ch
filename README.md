YOLOv8-4ch
==========
# DEMO
https://user-images.githubusercontent.com/104181368/222668654-0efb9323-4e6a-408c-90a1-a6807b9d8e40.mp4

| Best Model | Device          | Format        | Speed/Image | FPS  | AP@All |
| ---------- | --------------- | ------------- | ----------- | ---- | ------ |
| YOLOv8s    | RTX3090         | PyTorch       | 2.7 ms      | 370  | 0.842  |
|            |                 | TensorRT      | 2.6 ms      | 384  | -      |
|            |                 | TensorRT FP16 | 1.1 ms      | 909  | 0.840  |
|            | Jetson AGX Orin | PyTorch       | 8.9 ms      | 112  | -      |
|            |                 | TensorRT      | 3.0 ms      | 333  | -      |
|            | Intel i7-12700  | PyTorch       | 189.5 ms    | 5.2  | -      |
|            |                 | OpenVINO      | 63.1 ms     | 15.8 | -      |
|            |                 | ONNX          | 47.5 ms     | 21.0 | -      |

# Original
 This is forked repo from (ultralytics/ultralytics) on commit 74e4c94.  
 Catch up to changes [here](https://github.com/ultralytics/ultralytics/compare/74e4c94...main).

# Features
YOLOv8 をRGB-FIR向けに拡張したもの. 次の機能をオリジナルから追加している.
- [x] 1ch, 2ch, 3ch, 4chの学習・テストに対応
- [x] 4chの推論に対応
- [x] TensorRT, ONNX, OpenVINOのエクスポートに対応
- [x] データセットの読み込みを確認、cfgの設定をできるツール `check_dataset.py`
- [x] テスト結果をsvg, csvで保存
- [x] テスト結果の画像を全て書き出し

# TODO
基本的にはissueに投げる.  
開発をする際には自分のブランチを作成する. その後mainへマージする際には各ブランチ上でpull rebaseした後にPRを投げる. 
```bash
git pull --rebase origin main
```

# Installation
0. 必要に応じてDockerのセットアップやNVIDIA環境を入れる. [[参考]](https://github.com/Rits-Fujinolab/Docker-setup/blob/master/server.md)
1. cloneする. git環境がある人はOrganizationsにアクセス権限のあるuser名とmailを設定して
  ```bash
  git clone git@github.com:Rits-Fujinolab/YOLOv3-4ch.git
  ```
  または [GitHub CLI](https://cli.github.com) をインストールしてログイン認証後
  ```bash
  gh repo clone Rits-Fujinolab/YOLOv3-4ch
  ```

2. yolov3-4chのフォルダを開きdocker imageをbuildする. 容量は13GBくらいなのでそこそこ時間がかかる.
  ```bash
  ./tools.sh -b
  ```

3. データセットをdatasetフォルダーに入れる. [dataloader](utils/datasets.py) を魔改造してるため次のようなディレクトリ構造推奨...  
  シンボリックリンクでも認識可能なのでデータフォルダを作った後, フォルダごとにリンクを作るとスペースを節約できる.
  ```
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
  │       ├── FIR
  │       ├── FIR_labels
  │       ├── RGB
  │       ├── RGB_crop
  │       └── RGB_raw
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
      │   
      └── val
          ├── set06
          │   ├──V000
          │   │  ├── labels
          │   │  ├── person
          │   │  ├── lwir
          │   │  └── visible
          │   └──V001
          │      ├── labels
          │      ├── person
          │      ├── lwir
          │      └── visible
          └── set07
  ```

# Usage
## 通常モード
  ### コンテナ起動
  次のコマンドを実行.
  ```bash
  ./tools.sh -r
  ```

  ### データセットの準備
  [ここ](data/fujinolab-all.yaml)を参考にディレクトリとクラスを指定.  
  なおRGB画像、FIR画像、ラベルファイルは全て同じ名前を持っている必要がある (RGBを基準に出ディレクトリを置換している)

  ### 訓練
  [ここ](memo.txt)を参照. 基本的にはdataオプションとbatch-sizeオプション, epochsオプションで変更すればよい.  
  以下例...
  ```bash
  yolo detect train data=[yamlへのパス] model=[yamlまたはptへのパス]
  ```

  ### テスト
  [ここ](memo.txt)を参照. 基本的にはdataオプションと重みファイルオプションで変更すればよい. スライド用の画像を探す場合はsave-allオプションが便利.  
  以下例...
  ```bash
  yolo detect val model=[重みファイルへのパス] data=[データyamlへのパス] save-all
  ```

  ### 検知
  ```bash
  yolo detect predict model=[重みファイルへのパス] source=[データフォルダ] [検知結果画像を出力する場合は"save"]
  ```

  ## モデル変換
  NVIDIAデバイス向けのTensorRTやIntelデバイス向けのOpenVINO, オープンソース(主にAMDデバイス向け)のONNXへの変換を行うことができる.
  ```bash
  yolo export model=[重みファイルへのパス] format=[フォーマット]
  ```
  なおformatは[engine, openvino, onnx, tflite]など [[詳細](https://docs.ultralytics.com/modes/export/)].
## デバッグモード
  ### Dockerイメージをビルド
  次のコマンドを実行.
  ```bash
  ./tools.sh -c
  ```
  ### コンテナ起動
  次のコマンドを実行.
  ```bash
  ./tools.sh -d
  ```

  ### データセットの準備
  [ここ](data/fujinolab-all.yaml)を参考にディレクトリとクラスを指定.  
  なおRGB画像、FIR画像、ラベルファイルは全て同じ名前を持っている必要がある (RGBを基準に出ディレクトリを置換している)

  ### 一連のテスト
  [ここ](memo.txt)を参照. 基本的にはdataオプションとbatch-sizeオプション, epochsオプションで変更すればよい.  
  場合によってはdataやモデル名を指定して変更すること. test時には2000枚ほどを使用して訓練が行われる.
  以下例...
  ```bash
  yolo cfg=cfg/test.yaml
  ```
