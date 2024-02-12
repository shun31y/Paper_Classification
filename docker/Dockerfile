# 基本イメージの指定
FROM python:3.8-slim

# 必要なパッケージのインストール
RUN apt-get update 
RUN pip install jupyterlab
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy
RUN pip install feedparser
RUN pip install flake8
RUN pip install black
RUN pip install isort
# Jupyter Labの起動コマンド設定
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]