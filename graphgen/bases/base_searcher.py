from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseSearcher(ABC):
    """
    Abstract base class for searching and retrieving data.
    """

    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for data based on the given query.

        :param query: The searcher query.
        :param kwargs: Additional keyword arguments for the searcher.
        :return: List of dictionaries containing the searcher results.
        """
