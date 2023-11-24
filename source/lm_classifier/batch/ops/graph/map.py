import networkx as nx


from typing import Callable
from typing import Dict

from source.lm_classifier.batch.ops.functions import arguments_name


def map_nodes(graph: nx.DiGraph, f: Callable[..., Dict]) -> nx.DiGraph:
    nodes = graph.nodes_iter(data=True)

    args_required = arguments_name(f)
    args = ({'graph': graph, 'node': node, 'data': data} for node, data in nodes)
    args = ({k: v for k, v in a.items() if k in args_required} for a in args)

    for x in args:
        f(**x)

    return graph