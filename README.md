YOLOv3-4ch
==========

# Features
YOLOv3 をRGB-FIR向けに拡張したもの. 次の機能をオリジナルから追加している.
- 1ch, 2ch, 3ch, 4chの学習・テストに対応
- 4chの推論に対応
- データセットの読み込みを確認できるツール `check_dataset.py`
- テスト結果をsvg, csvで保存
- テスト結果の画像を全て書き出し

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
  <dataset>
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
  │   └── 20180903_1113  <-シンボリックリンク
  │       ├── FIR
  │       ├── FIR_labels
  │       ├── RGB
  │       ├── RGB_crop
  │       └── RGB_raw
  │   
  └── kaist-all
      ├── train
      │   ├── FIR
      │   ├── FIR_labels
      │   └── RGB
      └── val
          ├── FIR
          ├── FIR_labels
          ├── RGB
          └── RGB_labels
  ```

4. 必要に応じてオンラインで学習状況を確認できる [wandb](https://wandb.ai/home) に登録してログインキーを登録する. 詳細は[公式レポ](https://github.com/ultralytics/yolov5/issues/1289)参照.  
今まで通りtensor boradを使うなら[次の起動時](#起動)に`wandb off`をターミナルで実行. また [tools.sh](tools.sh) にてwandb関連のマウントを削除.  
削除: ~~`--mount type=bind,source="$(pwd)"/wandb,target=/usr/src/app/wandb \`~~  
削除: ~~`--mount type=bind,source=${HOME}${USERPROFILE}/.netrc,target=/root/.netrc \`~~

# Usage
## 通常モード
  ### 起動
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
  python train.py --data [データyamlへのパス] --hyps [パラメータyamlへのパス] --batch-size [n (自動推定:-1)] --epochs [エポック数]
  ```

  ### テスト
  [ここ](memo.txt)を参照. 基本的にはdataオプションと重みファイルオプションで変更すればよい. スライド用の画像を探す場合はsave-allオプションが便利.  
  以下例...
  ```bash
  python val.py --weights [重みファイルへのパス] --data [データyamlへのパス] --save-all
  ```

  ### 検知
  準備中

## デバッグモード
準備中
