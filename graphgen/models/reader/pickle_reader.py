import pickle
from typing import List, Union

import pandas as pd
import ray
from ray.data import Dataset

from graphgen.bases.base_reader import BaseReader
from graphgen.utils import logger


class PickleReader(BaseReader):
    """
    Read pickle files, requiring the schema to be restored to List[Dict[str, Any]].
    Each pickle file should contain a list of dictionaries with at least:
    - type: The type of the document (e.g., "text", "image", etc.)
    - if type is "text", "content" column must be present.

    Note: Uses ray.data.read_binary_files as ray.data.read_pickle is not available.
    For Ray >= 2.5, consider using read_pickle if available in your version.
    """

    def read(
        self,
        input_path: Union[str, List[str]],
    ) -> Dataset:
        """
        Read Pickle files using Ray Data.

        :param input_path: Path to pickle file or list of pickle files.
        :return: Ray Dataset containing validated documents.
        """
        if not ray.is_initialized():
            ray.init()

        # Use read_binary_files as a reliable alternative to read_pickle
        ds = ray.data.read_binary_files(input_path, include_paths=True)

        # Deserialize pickle files and flatten into individual records
        def deserialize_batch(batch: pd.DataFrame) -> pd.DataFrame:
            all_records = []
            for _, row in batch.iterrows():
                try:
                    # Load pickle data from bytes
                    data = pickle.loads(row["bytes"])

                    # Validate structure
                    if not isinstance(data, list):
                        logger.error(
                            "Pickle file {row['path']} must contain a list, got {type(data)}"
                        )
                        continue

                    if not all(isinstance(item, dict) for item in data):
                        logger.error(
                            "Pickle file {row['path']} must contain a list of dictionaries"
                        )
                        continue

                    # Flatten: each dict in the list becomes a separate row
                    all_records.extend(data)
                except Exception as e:
                    logger.error(
                        "Failed to deserialize pickle file %s: %s", row["path"], str(e)
                    )
                    continue

            return pd.DataFrame(all_records)

        # Apply deserialization and flattening
        ds = ds.map_batches(deserialize_batch, batch_format="pandas")

        # Validate the schema
        ds = ds.map_batches(self._validate_batch, batch_format="pandas")

        # Filter valid items
        ds = ds.filter(self._should_keep_item)
        return ds
