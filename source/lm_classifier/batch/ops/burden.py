from typing import Optional
from typing import List
from typing import Dict
from typing import Tuple

from collections import defaultdict
from functools import reduce

DataItem = Tuple[str, str, List[str], int, List[float]]


def _make_burden_instance(node_ids, prediction, probabilities, idx_to_label: Dict) -> Dict:
    """
    Creates a burden instance object
    """

    probabilities_ = []

    for ps in probabilities:
        probabilities_.append({
            idx_to_label[i]: p
            for i, p in enumerate(ps)
        })

    d = {
        'node-ids': node_ids,
        'class': idx_to_label[prediction],
        'probabilities': probabilities_
    }

    return d


def document_burden_view(data: List[DataItem], idx_to_label: Dict, bridges: Optional[List[str]] = None, ignore: Optional[List[str]] = None):
    """
    Creates and aggregates burden instances at the document level

    * Each burden instance contains and ordered list of the node ids
      that belong to the given instance;
    * The aggregation cannot be done across articles
    * Given two adjacent blocks, a and b, these are aggregated whenever one of the following is true:
        - They have the same burden class;
        - Block b class in `bridges`

    `bridges` is a list of classes indicating which classes can be used to merge two
    blocks into an individual burden instance;

    'ignore` is a list of classes that are not expected to be the head of a new burden instance
    these are ignored as burden initiators
    """

    if bridges is None:
        bridges = []

    if ignore is None:
        ignore = []

    label_to_idx = {v: k for k, v in idx_to_label.items()}
    bridges_ids = {label_to_idx[l] for l in bridges}
    ignore_ids = {label_to_idx[l] for l in ignore}

    articles = defaultdict(lambda: [])

    for article_id, block_id, text, pred, probs in data:
        articles[article_id].append((block_id, pred, probs))

    def reducer(acc, x):
        items, last_pred = acc
        block_id, pred, probs = x

        if not items:
            if pred not in ignore_ids:
                new_item = ([block_id], pred, [probs])
                items.append(new_item)
                return items, pred
            else:
                return acc

        head = items[-1]

        if (head[1] == pred) and (last_pred not in ignore_ids):
            head[0].append(block_id)
            head[2].append(probs)
            return items, pred

        elif pred in bridges_ids:
            head[0].append(block_id)
            head[2].append(probs)
            return items, pred

        elif pred not in ignore_ids:
            new_item = ([block_id], pred, [probs])
            items.append(new_item)
            return items, pred

        else:
            return acc

    instances = [reduce(reducer, blocks, ([], None))[0] for _, blocks in articles.items()]
    instances = sum(instances, [])
    instances = [_make_burden_instance(*(xs + (idx_to_label,))) for xs in instances]
    return instances
