import json
import logging
import networkx as nx

from typing import Dict

logger = logging.getLogger(__name__)

WM = Dict


def from_data(data: WM) -> nx.DiGraph:
    """
    Parses a dict representation back to networkx
    """
    graph = nx.DiGraph()

    for node_all_data in data["nodes"]:
        node_key = node_all_data["key"]
        graph.add_node(node_key, attr_dict=node_all_data["content"])

    for node_all_data in data["nodes"]:
        key = node_all_data["key"]
        graph.add_edges_from(map(lambda p: (p, key), node_all_data["predecessors"]))
        graph.add_edges_from(map(lambda s: (key, s), node_all_data["successors"]))

    return graph


def to_data(graph: nx.DiGraph) -> WM:
    """
    Parses a graph back to a WM object
    """
    result = {"document_name": graph.node['ROOT [0]']["meta"], "nodes": []}

    for node, data in graph.nodes(data=True):

        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))

        node = {
            "key": node,
            "content": data,
            "successors": successors,
            "predecessors": predecessors
        }

        result["nodes"].append(node)

    return result