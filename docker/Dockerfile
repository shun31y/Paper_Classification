# 基本イメージの指定
FROM python:3.8-slim

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y gcc python3-dev
COPY requirements.txt /work/requirements.txt
RUN pip install -r /work/requirements.txt
RUN pip install --upgrade jupyter ipywidgets
# Jupyter Labの起動コマンド設定
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''","--port=8000"]