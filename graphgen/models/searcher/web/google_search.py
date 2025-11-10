import requests
from fastapi import HTTPException

from graphgen.utils import logger

GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"


class GoogleSearch:
    def __init__(self, subscription_key: str, cx: str):
        """
        Initialize the Google Search client with the subscription key and custom searcher engine ID.
        :param subscription_key: Your Google API subscription key.
        :param cx: Your custom searcher engine ID.
        """
        self.subscription_key = subscription_key
        self.cx = cx

    def search(self, query: str, num_results: int = 1):
        """
        Search with Google and return the contexts.
        :param query: The searcher query.
        :param num_results: The number of results to return.
        :return: A list of searcher results.
        """
        params = {
            "key": self.subscription_key,
            "cx": self.cx,
            "q": query,
            "num": num_results,
        }
        response = requests.get(GOOGLE_SEARCH_ENDPOINT, params=params, timeout=10)
        if not response.ok:
            logger.error("Search engine error: %s", response.text)
            raise HTTPException(response.status_code, "Search engine error.")
        json_content = response.json()
        try:
            contexts = json_content["items"][:num_results]
        except KeyError:
            logger.error("Error encountered: %s", json_content)
            return []
        return contexts
