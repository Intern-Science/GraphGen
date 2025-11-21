from typing import List, Union

import ray
from ray.data import Dataset

from graphgen.bases.base_reader import BaseReader


class CSVReader(BaseReader):
    """
    Reader for CSV files.
    Columns:
        - type: The type of the document (e.g., "text", "image", etc.)
        - if type is "text", "content" column must be present.
    """

    def read(
        self,
        input_path: Union[str, List[str]],
        override_num_blocks: int = None,
    ) -> Dataset:
        """
        Read CSV files and return Ray Dataset.

        :param input_path: Path to CSV file or list of CSV files.
        :param override_num_blocks: Number of blocks for Ray Dataset reading.
        :return: Ray Dataset containing validated and filtered data.
        """

        ds = ray.data.read_csv(input_path, override_num_blocks=override_num_blocks)
        ds = ds.map_batches(self._validate_batch, batch_format="pandas")
        ds = ds.filter(self._should_keep_item)
        return ds
