import networkx as nx

from typing import List


def dfs_text(g: nx.DiGraph, n) -> List[str]:
    """
    Starting from `n` returns all of the text
    of that seaction in "reading order"

    e.g. all text from the document in 'reading order'
    >>> text = dfs_text(g, 'ROOT [0]')
    """
    nodes = [b for _, b in nx.dfs_edges(g, n)]
    text = sum([g.node[n]['text'] for n in nodes], [])
    text = g.node[n]['text'] + text
    text = [t.strip() for t in text]
    return text
