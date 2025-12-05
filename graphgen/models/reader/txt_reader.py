from typing import List, Union

import ray
from ray.data import Dataset

from graphgen.bases.base_reader import BaseReader


class TXTReader(BaseReader):
    def read(
        self,
        input_path: Union[str, List[str]],
    ) -> Dataset:
        """
        Read text files from the specified input path.
        :param input_path: Path to the input text file or list of text files.
        :return: Ray Dataset containing the read text data.
        """
        docs_ds = ray.data.read_text(input_path, encoding="utf-8")

        docs_ds = docs_ds.map(
            lambda row: {
                "type": "text",
                self.text_column: row["text"],
            }
        )

        docs_ds = docs_ds.filter(self._should_keep_item)
        return docs_ds
