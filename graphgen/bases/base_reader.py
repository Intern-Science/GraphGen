import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import requests
from ray.data import Dataset


class BaseReader(ABC):
    """
    Abstract base class for reading and processing data.
    """

    def __init__(self, text_column: str = "content"):
        self.text_column = text_column

    @abstractmethod
    def read(self, input_path: Union[str, List[str]]) -> Dataset:
        """
        Read data from the specified file path.

        :param input_path: Path to the input file or list of file paths.
        :return: Ray Dataset containing the read data.
        """

    def _should_keep_item(self, item: Dict[str, Any]) -> bool:
        """
        Determine whether to keep the given item based on the text column.

        :param item: Dictionary representing a data entry.
        :return: True if the item should be kept, False otherwise.
        """
        item_type = item.get("type")
        assert item_type in [
            "text",
            "image",
            "table",
            "equation",
            "protein",
        ], f"Unsupported item type: {item_type}"
        if item_type == "text":
            content = item.get(self.text_column, "").strip()
            return bool(content)
        return True

    @staticmethod
    def _image_exists(path_or_url: str, timeout: int = 3) -> bool:
        """
        Check if an image exists at the given local path or URL.
        :param path_or_url: Local file path or remote URL of the image.
        :param timeout: Timeout for remote URL requests in seconds.
        :return: True if the image exists, False otherwise.
        """
        if not path_or_url:
            return False
        if not path_or_url.startswith(("http://", "https://", "ftp://")):
            path = path_or_url.replace("file://", "", 1)
            path = os.path.abspath(path)
            return os.path.isfile(path)
        try:
            resp = requests.head(path_or_url, allow_redirects=True, timeout=timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False
