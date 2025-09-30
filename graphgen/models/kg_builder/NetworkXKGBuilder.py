from dataclasses import dataclass

from graphgen.bases import BaseKGBuilder


@dataclass
class NetworkXKGBuilder(BaseKGBuilder):
    def build(self, chunks):
        pass

    async def extract_all(self, chunks):
        pass

    async def extract(self, chunk):
        pass

    async def merge_nodes(self, nodes_data, kg_instance, llm):
        pass
