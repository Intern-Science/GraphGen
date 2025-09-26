from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseReader(ABC):
    """
    Abstract base class for reading and processing data.
    """

    def __init__(self, text_column: str = "content"):
        self.text_column = text_column

    @abstractmethod
    def read(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read data from the specified file path.

        :param file_path: Path to the input file.
        :return: List of dictionaries containing the data.
        """
