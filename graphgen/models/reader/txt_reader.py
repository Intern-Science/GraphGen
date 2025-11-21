from typing import List, Union

import ray
from ray.data import Dataset

from graphgen.bases.base_reader import BaseReader


class TXTReader(BaseReader):
    def read(
        self,
        input_path: Union[str, List[str]],
        override_num_blocks: int = 4,
    ) -> Dataset:
        """
        Read text files from the specified input path.
        :param input_path: Path to the input text file or list of text files.
        :param override_num_blocks: Number of blocks to override for Ray Dataset reading.
        :return: Ray Dataset containing the read text data.
        """
        docs_ds = ray.data.read_text(
            input_path, encoding="utf-8", override_num_blocks=override_num_blocks
        )

        docs_ds = docs_ds.map(
            lambda row: {
                "type": "text",
                self.text_column: row["text"],
            }
        )

        docs_ds = docs_ds.filter(self._should_keep_item)
        return docs_ds
