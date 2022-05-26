import os
import re
import json
import numpy as np
from ipycytoscape import CytoscapeWidget
from collections import defaultdict
import networkx as nx

from pybkb.utils.analysis import make_rv_level_nx

DEFAULT_STYLE_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'assets/bkb-style.json',
        )

def render_nx(
        G,
        style_file:str=None,
        layout_name:str='dagre',
        node_spacing:int=10,
        edge_length_val:int=10,
        include_source_weight:bool=True,
        subgraph_on:list=None,
        subgraph_on_partial:list=None,
        ):
    # Load style file
    if style_file is None:
        style_file = DEFAULT_STYLE_FILE
    with open(style_file, 'r') as json_file:
        style_obj = json.load(json_file)
    # Create widget
    cyto_obj = CytoscapeWidget()
    # Set layout
    cyto_obj.set_layout(name=layout_name, nodeSpacing=10, edgeLengthVal=10)
    # Set style
    cyto_obj.set_style(style_obj[0]['style'])
    if subgraph_on is not None:
        G = nx.subgraph(G, subgraph_on)
    if subgraph_on_partial is not None:
        # Gather nodes that match partial string
        nodes = set()
        for partial in subgraph_on_partial:
            matches = [n for n in G.nodes if partial in n]
            nodes.update(set(matches))
        # Subgraph
        G = nx.subgraph(G, list(nodes))
    # Add graph
    cyto_obj.graph.add_graph_from_networkx(G, directed=True)
    # Render
    return cyto_obj


def render(
        bkb,
        style_file:str=None,
        layout_name:str='dagre',
        node_spacing:int=10,
        edge_length_val:int=10,
        inode_padding:int=7,
        hide_sources:bool=False,
        rv_only:bool=False,
        include_source_weight:bool=True,
        subgraph_on:list=None,
        subgraph_on_partial:list=None,
        remove_prior_snodes:bool=False,
        ):
    """ Will render the BKB in a cytoscape ipython Widget. To be used in combination 
        with jupyter notebooks.

    Args:
        :param bkb: The BKB object to be rendered.
        :type bkb: pybkb.bkb.BKB
    
    Kwargs:
        :param style_file: Path to cyptoscape json file that contains the desired graph style.
        :type: str
        :param layout_name: The layout name corresponding to a cytoscape graph layout.
        :type layout_name: str
        :param node_spacing: The amount of spacing that should be used to separate nodes in the graph.
        :type node_spacing: int
        :param edge_length_val: The minimum amount of edge length that should initially appear in the graph.
        :type edge_length_val: int
        :param inode_padding: Amount of padding to use from I-node label to I-node box border.
        :type inode_padding: int
        :param hide_sources: Will hide source nodes from the graph.
        :type hide_sources: bool
        :param rv_only: Only plots the relationships between random variables. Helpful for graph analysis.
        :type rv_only: bool

    Returns:
        :returns: A cytoscape ipython widget.
        :type: ipycytoscape.CytoscapeWidget
    """
    # Load style file
    if style_file is None:
        style_file = DEFAULT_STYLE_FILE
    with open(style_file, 'r') as json_file:
        style_obj = json.load(json_file)
    # Create widget
    cyto_obj = CytoscapeWidget()
    # Set layout
    cyto_obj.set_layout(name=layout_name, nodeSpacing=10, edgeLengthVal=10)
    # Set style
    cyto_obj.set_style(style_obj[0]['style'])
    # Get Cytoscape data
    if not rv_only:
        data = build_bkb_cytoscape_data(
                bkb,
                inode_padding=inode_padding,
                hide_sources=hide_sources,
                subgraph_on=subgraph_on,
                subgraph_on_partial=subgraph_on_partial,
                remove_prior_snodes=remove_prior_snodes,
                )
        # Add graph
        cyto_obj.graph.add_graph_from_json(data["elements"])
    else:
        G = make_rv_level_nx(bkb, include_sources=not hide_sources, include_source_weight=include_source_weight) 
        if subgraph_on is not None:
            G = nx.subgraph(G, subgraph_on)
        if subgraph_on_partial is not None:
            # Gather nodes that match partial string
            nodes = set()
            for partial in subgraph_on_partial:
                matches = [n for n in G.nodes if partial in n]
                nodes.update(set(matches))
        # Add graph
        cyto_obj.graph.add_graph_from_networkx(G, directed=True)
    # Render
    return cyto_obj

def build_bkb_cytoscape_data(
        bkb,
        inode_padding:int=7,
        filepath:str=None,
        hide_sources:bool=False,
        subgraph_on:list=None,
        subgraph_on_partial:list=None,
        remove_prior_snodes:bool=False,
        ):
    """ Builds a cytoscape json (dict) representation of a BKB.

    Args:
        :param bkb: The bkb object for which to generate cytoscape data dictionary.
        :type bkb: pybkb.bkb.BKB

    Kwargs:
        :param inode_padding: Padding of the I-node box border around the I-node label.
        :type: inode_padding: int
        :param filepath: Optional filepath where to write json data.
        :type: str
        :param inode_padding: Amount of padding to use from I-node label to I-node box border.
        :type inode_padding: int
        :param hide_sources: Will hide source nodes from the graph.
        :type hide_sources: bool

    Returns:
        :returns: A json cytoscape graph representation
        :rtype: dict
    """
    # Basic data representation for cytoscape see: http://js.cytoscape.org/#notation/elements-json
    cyto = {
            "elements": {
                "nodes": [],
                "edges": [],
                }
            }
    # Encode all regular I-nodes
    if subgraph_on_partial is not None:
        # Gather nodes that match partial string
        subgraph_inodes = set()
        for partial in subgraph_on_partial:
            matches = [inode for inode in bkb.non_source_inodes if partial in inode[0]]
            subgraph_inodes.update(set(matches))
    else:
        subgraph_inodes = None
    filter_inodes = []
    for inode in bkb.non_source_inodes:
        if subgraph_on is not None:
            if inode not in subgraph_on and inode[0] not in subgraph_on:
                continue
        if subgraph_inodes is not None and inode not in subgraph_inodes:
            continue
        idx = bkb.inodes_indices_map[inode]
        feature, state = inode
        name = f'{feature} = {state}'
        cyto["elements"]["nodes"].append(
                {
                    "data": {
                        "id": name,
                        "name": name,
                        "type": "i",
                        "label_width": len(name) * inode_padding
                        }
                    }
                )
        filter_inodes.append(inode)
    # Flag that might be used later to determine if this is a collapsed or fused BKB
    is_collapsed = False
    src_feature_id_map = {}
    src_inode_id_map = {}
    src_collection_map = {}
    # Encode all source I-nodes
    if not hide_sources:
        for src_feature_idx, src_feature in enumerate(bkb.source_features):
            # Create the Source Feature Node
            src_feature_name = re.split('\[|\]', src_feature)[1]
            name = f'Source({src_feature_name})'
            src_node_id = name #f'sn{src_feature_idx}'
            src_feature_id_map[src_feature] = src_node_id
            # Add node to cytoscape
            cyto["elements"]["nodes"].append(
                    {
                        "data": {
                            "id": src_node_id,
                            "name": name,
                            "type": 'src_f',
                            "label_width": len(name) * inode_padding
                            }
                        }
                    )
            # Collect all source feature states
            src_states = []
            for src_state in bkb.inodes_map[src_feature]:
                # Collect source collection states
                if type(src_state) == frozenset:
                    # Create a Source collection node
                    src_collection_node_id = f'{src_feature}-{src_state}'
                    src_collection_map[(src_feature, src_state)] = src_collection_node_id
                    cyto["elements"]["nodes"].append(
                            {
                                "data": {
                                    "id": src_collection_node_id,
                                    "name": '',
                                    "type": 'src_c',
                                    "parent": src_node_id,
                                    "label_width": len(name) * inode_padding
                                    }
                                }
                            )
                    is_collapsed = True
                    for i, _state in enumerate(src_state):
                        # Make a src inode and attach it to the source collection
                        cyto["elements"]["nodes"].append(
                                {
                                    "data": {
                                        "id": _state, #f'{src_collection_node_id}-{i}',
                                        "name": _state,
                                        "type": 'src_i',
                                        "parent": src_collection_node_id,
                                        "label_width": len(_state) * inode_padding
                                        }
                                    }
                                )
                # Or collect normal fused source states
                else:
                    src_states.append(src_state)
            # Now add in source nodes
            for src_state_idx, src_state in enumerate(src_states):
                # Add node to cytoscape
                src_inode_id = src_state #f'{src_node_id}-{src_state_idx}'
                src_inode_id_map[(src_feature, src_state)] = src_inode_id
                cyto["elements"]["nodes"].append(
                        {
                            "data": {
                                "id": src_inode_id,
                                "name": src_state,
                                "type": 'src_i',
                                "label_width": len(src_state) * inode_padding,
                                "parent": src_node_id,
                                }
                            }
                        )
    # Encode the S-nodes
    edge_id = 0
    for snode_idx, snode_prob in enumerate(bkb.snode_probs):
        if snode_prob == 0:
            continue
        head = bkb.get_snode_head(snode_idx)
        if '__Source__' in head[0] and hide_sources:
            continue
        if '__Source__' not in head[0] and head not in filter_inodes:
            continue
        src_node_id = src_feature_id_map.get(head[0], None)
        src_inode_id = src_inode_id_map.get(head, None)
        src_collection_id = src_collection_map.get(head, None)
        snode_id = f's{snode_idx} = {np.round(snode_prob, 3)}'
        # Check if this is a source head inode
        if src_node_id is not None:
            if src_collection_id is not None:
                # We will make a single prior S-node going into the group
                cyto["elements"]["nodes"].append(
                        {
                            "data": {
                                "id": snode_id, #f's{snode_idx}',
                                "name": snode_id, #str(snode_prob),
                                "type": "s",
                                "parent": src_node_id,
                                }
                            }
                        )
                # Add head edge
                cyto["elements"]["edges"].append(
                        {
                            "data": {
                                "id": f'e{edge_id}',
                                "source": snode_id, #f's{snode_idx}',
                                "target": src_collection_id,
                                }
                            }
                        )
            else:
                # We will make a prior S-node that lives in the src group compound node.
                cyto["elements"]["nodes"].append(
                        {
                            "data": {
                                "id": snode_id, #f's{snode_idx}',
                                "name": snode_id, #str(snode_prob),
                                "type": "s",
                                "parent": src_node_id
                                }
                            }
                        )
                # Add head edge
                cyto["elements"]["edges"].append(
                        {
                            "data": {
                                "id": f'e{edge_id}',
                                "source": snode_id, #f's{snode_idx}',
                                "target": src_inode_id,
                                }
                            }
                        )
            edge_id += 1
            # We can just continue since a source I-node head won't have a tail.
            continue
        # Get head and tail indices
        head_idx = bkb.inodes_indices_map[head]
        head_id = f'{head[0]} = {head[1]}'
        tail = bkb.get_snode_tail(snode_idx)
        # Collect Tail IDs
        new_tail = []
        for tail_inode in tail:
            edge_id += 1
            # Check to see if this is a source inode
            if '__Source__' in tail_inode[0]:
                if src_inode_id_map.get(tail_inode, None) is not None:
                    tail_id = src_inode_id_map[tail_inode]
                elif src_collection_map.get(tail_inode, None) is not None:
                    tail_id = src_collection_map[tail_inode]
                elif hide_sources:
                    continue
            elif tail_inode not in filter_inodes:
                continue
            else:
                tail_id = f'{tail_inode[0]} = {tail_inode[1]}' #f'n{bkb.inodes_indices_map[tail_inode]}'
            new_tail.append(tail_id)
        if len(new_tail) == 0 and remove_prior_snodes:
            continue
        # Create S-Node
        cyto["elements"]["nodes"].append(
                {
                    "data": {
                        "id": snode_id, #f's{snode_idx}',
                        "name": snode_id, #str(snode_prob),
                        "type": "s",
                        }
                    }
                )
        # Add head edge
        cyto["elements"]["edges"].append(
                {
                    "data": {
                        "id": f'e{edge_id}',
                        "source": snode_id, #f's{snode_idx}',
                        "target": head_id, #f'n{head_idx}',
                        }
                    }
                )
        # Now make tail
        for tail_id in new_tail:
            edge_id += 1
            cyto["elements"]["edges"].append(
                    {
                        "data": {
                            "id": f'e{edge_id}',
                            "source": tail_id,
                            "target": snode_id #f's{snode_idx}',
                            }
                        }
                    )
    if filepath:
        with open(filepath, 'w') as json_file:
            json.dump(cyto, json_file, indent=2)
    return cyto


def build_bkb_cytoscape_data_rv_level(bkb, inode_padding:int=7, filepath:str=None, hide_sources:bool=False):
    """ Builds a cytoscape json (dict) representation of the BKB at the RV level. Helpful for visualization.

    Args:
        :param bkb: The bkb object for which to generate cytoscape data dictionary.
        :type bkb: pybkb.bkb.BKB

    Kwargs:
        :param inode_padding: Padding of the I-node box border around the I-node label.
        :type: inode_padding: int
        :param filepath: Optional filepath where to write json data.
        :type: str
        :param inode_padding: Amount of padding to use from I-node label to I-node box border.
        :type inode_padding: int
        :param hide_sources: Will hide source nodes from the graph.
        :type hide_sources: bool

    Returns:
        :returns: A json cytoscape graph representation
        :rtype: dict
    """
    # Basic data representation for cytoscape see: http://js.cytoscape.org/#notation/elements-json
    cyto = {
            "elements": {
                "nodes": [],
                "edges": [],
                }
            }
    # Encode all regular random variables
    feature_id_map = {}
    for idx, rv in enumerate(bkb.non_source_features):
        feature_id_map[rv] = idx
        name = f'{rv}'
        cyto["elements"]["nodes"].append(
                {
                    "data": {
                        "id": f'n{idx}',
                        "name": name,
                        "type": "i",
                        "label_width": len(name) * inode_padding
                        }
                    }
                )
    # Flag that might be used later to determine if this is a collapsed or fused BKB
    src_feature_id_map = {}
    # Encode all source I-nodes
    if not hide_sources:
        for src_feature_idx, src_feature in enumerate(bkb.source_features):
            # Create the Source Feature Node
            src_node_id = f'sn{src_feature_idx}'
            src_feature_id_map[src_feature] = src_node_id
            src_feature_name = re.split('\[|\]', src_feature)[1]
            name = f'Source({src_feature_name})'
            # Add node to cytoscape
            cyto["elements"]["nodes"].append(
                    {
                        "data": {
                            "id": src_node_id,
                            "name": name,
                            "type": 'src_f',
                            "label_width": len(name) * inode_padding
                            }
                        }
                    )
    # Encode the S-nodes
    edge_id = 0
    edges = defaultdict(set)
    for snode_idx, snode_prob in enumerate(bkb.snode_probs):
        if snode_prob == 0:
            continue
        head = bkb.get_snode_head(snode_idx)
        if '__Source__' in head[0]:
            continue
        # Get head and tail indices
        tail = bkb.get_snode_tail(snode_idx)
        # Create edge from each tail feature to the head feature, if there is not already an edge
        head_feature, _ = head
        head_idx = feature_id_map[head_feature]
        for tail_feature, _ in tail:
            if '__Source__' in tail_feature and hide_sources:
                continue
            elif '__Source__' in tail_feature:
                tail_id = src_feature_id_map[tail_feature]
            else:
                tail_id = f'n{feature_id_map[tail_feature]}'
            if tail_feature not in edges[head_feature]:
                # Add head edge
                cyto["elements"]["edges"].append(
                        {
                            "data": {
                                "id": f'e{edge_id}',
                                "source": tail_id,
                                "target": f'n{head_idx}',
                                }
                            }
                        )
                # Add edge to edges so we don't double add
                edges[head_feature].add(tail_feature)
            edge_id += 1
    if filepath:
        with open(filepath, 'w') as json_file:
            json.dump(cyto, json_file, indent=2)
    return cyto
