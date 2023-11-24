import networkx as nx

from typing import Dict
from typing import List
from typing import Tuple

from source.lm_classifier.batch.ops.graph.remove import irrelevant_block
from source.lm_classifier.batch.ops.graph.remove import remove_schedules
from source.lm_classifier.batch.ops.graph.remove import remove_irrelevant_nodes


def text_blocks(g: nx.DiGraph) -> List[str]:
    g = remove_irrelevant_nodes(g)
#     g = remove_schedules(g)

    articles = get_articles(g)
    article_titles = [g.node[a].get('title') for a in articles]

    blocks = [split_blocks(g, a) for a in articles]

    blocks = [[(title, b[0], b[1]) for b in block] for title, block in zip(article_titles, blocks)]
    blocks = sum(blocks, [])

    blocks = [(title, url, text) for title, url, text in blocks if not irrelevant_block(text)]
    blocks = [text for _, __, text in blocks]
    return blocks


def get_articles(g):
    # TODO: verify if "PBlock" should be included
    article_functional_tags = ['P1group', 'P', 'P2', 'P1', 'Pblock']
    articles_functional_containers = ['root', 'schedule', 'chapter', 'part', 'group']

    articles = [
        n for n, d in g.nodes(data=True)
        if d['meta'] in article_functional_tags
    ]

    articles = [
        a for a in articles if
        any((any(((i == g.node[m]['meta'].lower()) for i in articles_functional_containers)) for m, n in g.in_edges(a)))
    ]

    return articles


def dfs_text(g: nx.DiGraph, n) -> List[str]:
    """
    Starting from `n` returns all of the text
    of that seaction in "reading order"
    """
    nodes = [b for _, b in nx.dfs_edges(g, n)]
    text = sum([g.node[n]['text'] for n in nodes], [])
    text = g.node[n]['text'] + text
    text = [t.strip() for t in text]
    return text


def _p1_with_two_p3(g, n):
    nodes = [b for a, b in g.out_edges(n)]
    if len(nodes) > 2:
        return False
    else:
        return all((g.node[n]['meta'] in ['P3'] for n in nodes))


def split_blocks(g, article) -> List[Tuple[str, List[str]]]:
    """
    Given an article, segregate it into different blocks
    For each article returns a list of blocks
    [blocks] :: (node_id, [str])
    """
    out_nodes = [b for a, b in g.out_edges(article)]

    opening_text = g.node[article]['text']

    def default():
        id_ = next(
            (g.node[b]['id'] for a, b in ([(None, article)] + list(nx.dfs_edges(g, article))) if
             'gov.uk' in g.node[b]['id']),
            g.node[article]['id']
        )
        result = [(id_, dfs_text(g, article))]
        return result

    if any([
        len(out_nodes) == 0,
        len(out_nodes) == 1 and g.node[out_nodes[0]]['meta'] in ['P'],
        len(out_nodes) == 1 and (g.node[out_nodes[0]]['meta'] in ['P1']) and _p1_with_two_p3(g, out_nodes[0]),
        g.node[article]['meta'] == 'P'
    ]):
        result = default()

    elif len(out_nodes) == 1:

        nodes = [b for a, b in g.out_edges(out_nodes[0])]

        if len(nodes) <= 1:
            result = default()

        else:
            result = [(g.node[n]['id'], dfs_text(g, n)) for n in nodes]
            result[0] = (result[0][0], opening_text + result[0][1])

    else:
        result = [(g.node[n]['id'], dfs_text(g, n)) for n in out_nodes]
        result[0] = (result[0][0], opening_text + result[0][1])

    return result

