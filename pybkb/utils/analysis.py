import copy
import networkx as nx
from collections import defaultdict

def make_rv_level_dict(bkb, include_sources=False, include_source_weight=True, node_padding:int=7):
    # Initialize
    node_info_template = {"label_width": None}
    edge_info_template = {"snode_info": [], "weight": 1}
    nodes = defaultdict(lambda: copy.deepcopy(node_info_template))
    edges = defaultdict(lambda: copy.deepcopy(edge_info_template))
    snodes_by_head = bkb.snodes_by_head
    for snode_idx, snode_prob in enumerate(bkb.snode_probs):
        head_feature, head_state = bkb.get_snode_head(snode_idx)
        if '__Source__' in head_feature and not include_sources:
            continue
        nodes[head_feature]["label_width"] = len(head_feature) * node_padding
        tail_filter = []
        weight = None
        for tail_feature, tail_state in bkb.get_snode_tail(snode_idx):
            if '__Source__' in tail_feature:
                weight = bkb.snode_probs[snodes_by_head[(tail_feature, tail_state)][0]]
                if not include_sources:
                    continue
            nodes[tail_feature]["label_width"] = len(tail_feature) * node_padding
            tail_filter.append(tail_feature)
        snode_info = (
                (head_feature, head_state),
                tail_filter,
                snode_prob,
                )
        # Add edges
        for tail_feature in tail_filter:
            edges[(tail_feature, head_feature)]["snode_info"].append(snode_info)
            if weight is not None and include_source_weight:
                edges[(tail_feature, head_feature)]["weight"] = weight
    return nodes, edges

def make_rv_level_nx(bkb, include_sources=False, include_source_weight=True):
    nodes, edges = make_rv_level_dict(bkb, include_sources, include_source_weight)
    # Convert to network X multi directed graph
    G = nx.MultiDiGraph()
    # Add nodes
    for node_id, node_data in nodes.items():
        G.add_node(node_id, label_width = node_data["label_width"])
    for (u,v), data in edges.items():
        if include_source_weight:
            G.add_edge(u, v, snode_info=data["snode_info"], weight=data["weight"])
        else:
            G.add_edge(u, v, snode_info=data["snode_info"])
    return G
