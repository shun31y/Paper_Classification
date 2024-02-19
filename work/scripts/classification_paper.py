import os
import re
import sys

import openai
import pandas as pd
from bertopic import BERTopic
from bertopic.backend import OpenAIBackend
from bertopic.representation import OpenAI
from sklearn.cluster import KMeans
from pyclustering.cluster.center_initializer import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

sys.path.append("../utils")
from openai_embeddings import OpenAICustomBackend


class Classification_Topic:
    def __init__(
        self, openai_api_key: str, saved_file_name: str = "retrieved_paper.csv"
    ):

        self.api_key = openai_api_key
        self.saved_file_name = saved_file_name
        self.docs_df = pd.DataFrame()
        self.topics = []
        self.df_info = pd.DataFrame()

    def classification_papers_csv(self, upper_num_topics: int):
        # 　ドキュメントのデータフレームからアブストラクトのみを抽出
        self.docs_df = pd.read_csv("/work/data/" + self.saved_file_name)
        doc_list = list(self.docs_df["Abstruct"])
        # 　OpenAI系統のモデルの宣言
        client = openai.OpenAI(api_key=self.api_key)
        embedding_model = OpenAICustomBackend(client=client)
        prompt = """
            I have a topic that contains the following documents:
            [DOCUMENTS]
            The topic is described by the following keywords: [KEYWORDS]
            Based on the information above, extract a short but highly descriptive topic label of at most 5 words.
            The number of words should not exceed 100 characters for the entire label
            Make sure it is in the following format:
            topic: <topic label>
            """
        representation_model = OpenAI(
            client, model="gpt-3.5-turbo", chat=True, prompt=prompt
        )
        # 分類モデルの定義
        topic_model = BERTopic(
            verbose = True,
            lanugage = "english",
            calculate_probabilities = False,
            nr_topics="auto",
            embedding_model = embedding_model
        )
        
        # 文書の埋め込みベクトルを準備する
        doc_embeddings = topic_model._extract_embeddings(doc_list)

        #X-meansクラスタリングのインスタンスを初期化・作成・実行
        initial_centers = kmeans_plusplus_initializer(doc_embeddings, upper_num_topics).initialize()
        xmeans_instance = xmeans(doc_embeddings, initial_centers, ccore=True, kmax=upper_num_topics)
        xmeans_instance.process()

        # クラスタを取得
        self.df_info = xmeans_instance.get_clusters()

    def make_and_save_topic_csv(self, keyword):
        # トピック分類後はトピックのインデックスが振られているためトピック名に変更する
        self.docs_df["Topic"] = [
            self.df_info[self.df_info["Topic"] == num]["Name"].values[0]
            for num in self.df_info
        ]
        # 保存するディレクトリの作成
        dir = "/work/data/{}".format(keyword)
        if not os.path.exists(dir):
            os.makedirs(dir)
        for topic in self.df_info["Name"].unique():
            topic = re.sub(r"^\d+_", "", topic)
            topic_dir = os.path.join(dir, topic)
            if not os.path.exists(topic_dir):
                os.makedirs(topic_dir)
        # データフレームをトピックごとに分割
        docs_df_grouped = self.docs_df.groupby("Topic")
        dfs = {re.sub(r"^\d+_", "", topic): group for topic, group in docs_df_grouped}
        for topic in self.df_info["Name"].unique():
            topic = re.sub(r"^\d+_", "", topic)
            topic_file_path = os.path.join(dir, topic, "papers_data.csv")
            # 保存
            dfs[topic].to_csv(topic_file_path)
