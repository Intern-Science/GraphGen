from dataclasses import dataclass, fields

from graphgen.models.strategy.base_strategy import BaseStrategy


@dataclass
class TraverseStrategy(BaseStrategy):
    # 生成的QA形式：原子、多跳、聚合型
    qa_form: str = "atomic" # "atomic" or "multi_hop" or "aggregated"
    # 最大边数和最大token数方法中选择一个生效
    expand_method: str = "max_tokens" # "max_width" or "max_tokens"
    # 单向拓展还是双向拓展
    bidirectional: bool = True
    # 每个方向拓展的最大边数
    max_extra_edges: int = 5
    # 最长token数
    max_tokens: int = 256
    # 每个方向拓展的最大深度
    max_depth: int = 2
    # 同一层中选边的策略（如果是双向拓展，同一层指的是两边连接的边的集合）
    edge_sampling: str = "max_loss" # "max_loss" or "min_loss" or "random"
    # 孤立节点的处理策略
    isolated_node_strategy: str = "add" # "add" or "ignore"
    loss_strategy: str = "only_edge"  # only_edge, both

    def to_yaml(self):
        strategy_dict = {}
        for f in fields(self):
            strategy_dict[f.name] = getattr(self, f.name)
        return {"traverse_strategy": strategy_dict}
