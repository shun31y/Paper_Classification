# Paper_Classification
---
## 論文検索・トピック分類を行うライブラリ
## Quick Start

1. リポジトリをclone
   ```bash
   https://github.com/shun31y/Paper_Classification.git
   ```
2. dockerイメージを作成する。dockerディレクトリに移動して以下を行う
   ```bash
   docker build -t impimage .
   docker compose up -d
   ```
3. localホスト(localhost:8000/lab/tree/work/notebook/retrieve_and_classification.ipynb)にアクセスする.
4. .env.exampleにOpenAI_APIKEYを入力後以下コマンドを実行して.envファイルを作成する
   ```bash
   cd work/notebook
   cp .env.example .env
   ```
5. 全てのセルを実行する。論文は新しい順で検索される
---
## 実行例
<img width="409" alt="スクリーンショット 2024-02-13 23 12 39" src="https://github.com/shun31y/Paper_Classification/assets/145087663/5504fe5a-610c-43cf-a388-e1187f1a598c"><br>
キーワード:ViT、<br>
検索数1000件にしてみる<br>
<img width="487" alt="スクリーンショット 2024-02-13 23 12 52" src="https://github.com/shun31y/Paper_Classification/assets/145087663/dd661d95-d0fa-4e66-98cb-b45d6b2c0bad"><br>
最大クラスタ数は30で指定する。<br>
<img width="219" alt="スクリーンショット 2024-02-13 23 13 30" src="https://github.com/shun31y/Paper_Classification/assets/145087663/8e577b86-ad53-459d-ad34-1850de294f9f"><br>
全てのセルを実行後キーワード名のディレクトリが作成される<br>
<img width="499" alt="スクリーンショット 2024-02-13 23 13 48" src="https://github.com/shun31y/Paper_Classification/assets/145087663/93102bc3-f7ff-4f86-b664-6d31710096c3"><br>
キーワードディレクトリ以下にトピックごとのディレクトリが作成され内部に論文をまとめたCSVが作成される<br>




