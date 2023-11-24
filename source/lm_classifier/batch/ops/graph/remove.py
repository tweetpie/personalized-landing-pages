import networkx as nx


def remove_from(g, n):
    nodes = [n] + [b for _, b in nx.dfs_edges(g, n)]
    for n in nodes:
        g.remove_node(n)
    return g


def ignore_title(title):
    title = title.lower()

    keywords = [
        'repeal', 'additional conditions', 'interpretation',
        'power', 'introductory', 'index', 'preliminary',
        'section', 'this', 'overview',
        'meaning of', 'terms', 'exemptions', 'scope',
        'regulations under', 'changes to', 'commencement',
        'changes to', 'extent', 'transitional provision',
        'final provision', 'final conditions', 'disclosure',
        'powers in', 'moratorium', 'prohibition',
        'short title', 'crown application', 'excluded', 'removal',
        'right of', 'miscellaneous', 'review of', 'supplementary',
        'citation', 'signature', 'exploratory note', 'in part',
         'enforcement'
    ]

    result = any((i in title for i in keywords))
    return result


def remove_irrelevant_nodes(g):
    containers = ['part', 'group', 'pblock']

    nodes = [n for n in g.nodes() if not g.node[n]['meta'].lower() in containers]
    nodes_to_remove = [n for n in nodes if ignore_title(g.node[n].get('title', ''))]

    for n in nodes_to_remove:
        try:
            remove_from(g, n)
        except Exception as e:
            pass
    return g


def remove_schedules(g):
    """
    Removes every `Schedule` section
    from the graph
    """
    for n in g.nodes():
        try:
            if 'schedule' in g.node[n]['meta'].lower():
                remove_from(g, n)
        except Exception as e:
            pass
    return g


def irrelevant_block(text):
    text = '\n'.join(text)

    return any([
        '___________' in text
    ])