import requests
from fastapi import HTTPException

from graphgen.utils import logger

BING_SEARCH_V7_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
BING_MKT = "en-US"


class BingSearch:
    """
    Bing Search client to searcher with Bing.
    """

    def __init__(self, subscription_key: str):
        self.subscription_key = subscription_key

    def search(self, query: str, num_results: int = 1):
        """
        Search with Bing and return the contexts.
        :param query: The searcher query.
        :param num_results: The number of results to return.
        :return: A list of searcher results.
        """
        params = {"q": query, "mkt": BING_MKT, "count": num_results}
        response = requests.get(
            BING_SEARCH_V7_ENDPOINT,
            headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
            params=params,
            timeout=10,
        )
        if not response.ok:
            logger.error("Search engine error: %s", response.text)
            raise HTTPException(response.status_code, "Search engine error.")
        json_content = response.json()
        try:
            contexts = json_content["webPages"]["value"][:num_results]
        except KeyError:
            logger.error("Error encountered: %s", json_content)
            return []
        return contexts
