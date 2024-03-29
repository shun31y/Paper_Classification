import sys

import feedparser
import pandas as pd


class Retrieve_Arxiv:
    def __init__(self):

    def retrieve(self, keyword: str, max_num: int) -> None:
        """論文をarxivから検索、dfにするまで一括で行う関数

        Args:
            keyword (str): 検索したい論文に関連する単語
            max_num (int): 取得する論文の最大数
        """
        documents = self.send_requests(keyword, max_num)
        df = self.preprocess(documents)
        self.save_df_to_csv(df)

    def send_requests(self, keyword: str, max_num: int) -> list:
        """arxivにリクエストを送る関数

        Args:
            keyword (str): 検索したい論文に関連する単語
            max_num (int): 取得する論文の最大数

        Returns:
            list: 取得した論文についての情報(辞書型)を格納したリスト
        """
        request_message = self.__get_requests_message(keyword, max_num)
        d = feedparser.parse(request_message)
        return d["entries"]

    def preprocess(self, documents: list) -> pd.DataFrame:
        """取得したドキュメントデータから必要なデータを抜き出してdfにまとめる関数

        Args:
            documents (list): 取得した論文についての情報(辞書型)を格納したリスト

        Returns:
            pd.DataFrame: 取得した論文について必要な情報を抜き出したデータフレーム
        """
        properties = [
            [doc.title, doc.link, doc.published, doc.summary, doc.category]
            for doc in documents
        ]
        df = pd.DataFrame(
            properties, columns=["Title", "URL", "Date", "Abstruct", "Category"]
        )
        return df

    def __get_requests_message(self, keyword: str, max_num: int) -> str:
        """リクエストメッセージをコマンドライン引数から作成する内部関数

        Args:
            keyword (str): 検索したい論文に関連する単語
            max_num (int): 取得する論文の最大数

        Returns:
            str: リクエストするメッセージ
        """
        request_message = "http://export.arxiv.org/api/query?search_query={keyword}&max_results={max_num}&sortBy=lastUpdatedDate".format(
            keyword=keyword, max_num=max_num
        )
        return request_message


def main(keyword, max_num):
    re = Retrieve_Arxiv()
    re.retrieve(keyword, max_num)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <keyword> <max_num>")
        sys.exit(1)
    keyword = sys.argv[1]
    try:
        max_num = int(sys.argv[2])
    except ValueError:
        print("Error: <max_num> must be an integer.")
        sys.exit(1)
    main(keyword, max_num)
