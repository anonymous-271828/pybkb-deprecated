import networkx as nx
import matplotlib.pyplot as plt
from grave import plot_network
from grave.style import use_attributes
import logging
import itertools

#-- Setup Module Logger
logger = logging.getLogger('pybkb.common.bkb_grapher')
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
console.setFormatter(formatter)
logger.addHandler(console)

#-- To run test code uncomment this line.
#from bayesianKnowledgeBase import bayesianKnowledgeBase as BKB

def make_graph(bkb):
    graph = BkbGraph(bkb)
    return graph

def _process_node_name(node_name, num_char=30):
    if len(node_name) <= num_char:
        return node_name
    else:
        string = node_name[:(num_char - 3)] + '...'
        return string


class BkbGraph(object):
    def __init__(self, bkb):
        self.bkb = bkb

    def _build_bkb_graph(self):
        """ Builds the graph of the entire BKB as a NetworkZ DiGraph.
        """
        G = nx.DiGraph()
        i_nodes = [(comp_name, state_name) for comp_name, state_name in self.bkb.getINodeNames()]
        i_nodes_collected = set()
        snodes_by_head = self.bkb.constructSNodesByHead()
        _snode_graph_idx = len(i_nodes)
        _snode_label_idx = 0

        for head, snodes in snodes_by_head.items():
            #-- Process head
            head_comp_idx, head_state_idx = head
            head_comp_name = self.bkb.getComponentName(head_comp_idx)
            head_state_name = self.bkb.getComponentINodeName(head_comp_idx, head_state_idx)
            logger.debug('Processing I-node: {} = {}'.format(head_comp_name, head_state_name))
            #-- Sometimes it is the case that BKBs are not cleaned of headless inodes. So we do that here.
            if len(snodes) == 0:
                logger.info('I-node: {} = {} had no incoming s-nodes so it was not graphed')
                continue
            head_idx = i_nodes.index((head_comp_name, head_state_name))
            i_nodes_collected.add((head_comp_name, head_state_name))
            if head_idx not in G.nodes():
                i_node_name = '{}\n{}'.format(_process_node_name(head_comp_name),
                                              _process_node_name(head_state_name))
                G.add_node(head_idx, node_type='i', label=i_node_name)
            for snode in snodes:
                _snode_graph_idx += 1
                _snode_label_idx += 1
                logger.debug('Processing S-node with label: {}'.format(_snode_label_idx))
                s_node_name = r'$s_{%s} = %.3f$' % (str(_snode_label_idx), snode.getProbability())
                G.add_node(_snode_graph_idx, node_type='s', label=s_node_name)

                #-- Add edge from snode to head
                G.add_edge(_snode_graph_idx, head_idx)

                #-- Process snode tail
                for tail_idx in range(snode.getNumberTail()):
                    (tail_comp_idx, tail_state_idx) = snode.getTail(tail_idx)
                    tail_comp_name = self.bkb.getComponentName(tail_comp_idx)
                    tail_state_name = self.bkb.getComponentINodeName(tail_comp_idx, tail_state_idx)
                    tail_idx = i_nodes.index((tail_comp_name, tail_state_name))
                    if tail_idx not in G.nodes():
                        i_node_name = '{}\n{}'.format(_process_node_name(tail_comp_name),
                                                      _process_node_name(tail_state_name))
                        G.add_node(tail_idx, node_type='i', label=i_node_name)
                    G.add_edge(tail_idx, _snode_graph_idx)

        if len(set(i_nodes) - set(i_nodes_collected)) > 0:
            logger.warning('Not all I-nodes are graphed because they have no attached S nodes.')
        return G

    def _build_inference_graph(self, inference):
        """ Builds the inference graph given the BKB.

            :param inference: A dictionary return from the get_inference method in
                the reasoning_result API.
            :type inference: dict
        """
        # Extract inodes and snodes.
        inode_names = [(comp_name, state_name) for comp_name, state_name in self.bkb.getINodeNames()]
        '''
        inodes = inference["inodes"]
        inode_names = [
                (
                    self.bkb.getComponentName(comp_idx),
                    self.bkb.getComponentINodeName(comp_idx, state_idx)
                ) for comp_idx, state_idx in inodes ]
        print(inode_names)
        '''     
        snodes = inference["snodes_used"]
        _snode_graph_idx = len(inode_names)
        _snode_label_idx = 0

        # Build G
        G = nx.DiGraph()

        for snode in snodes:
            head_comp_idx, head_state_idx = snode.getHead()
            head_comp_name = self.bkb.getComponentName(head_comp_idx)
            head_state_name = self.bkb.getComponentINodeName(head_comp_idx, head_state_idx)

            # Add Head I-node to graph
            head_idx = inode_names.index((head_comp_name, head_state_name))
            if head_idx not in G.nodes():
                i_node_name = '{}\n{}'.format(_process_node_name(head_comp_name),
                                              _process_node_name(head_state_name))
                G.add_node(head_idx, node_type='i', label=i_node_name)

            # Add S-node
            s_node_name = r'$s_{%s} = %.3f$' % (str(_snode_label_idx), snode.getProbability())
            G.add_node(_snode_graph_idx, node_type='s', label=s_node_name)

            # Add head to S-node edge
            G.add_edge(_snode_graph_idx, head_idx)

            #-- Process snode tail
            for tail_idx in range(snode.getNumberTail()):
                (tail_comp_idx, tail_state_idx) = snode.getTail(tail_idx)
                tail_comp_name = self.bkb.getComponentName(tail_comp_idx)
                tail_state_name = self.bkb.getComponentINodeName(tail_comp_idx, tail_state_idx)
                tail_idx = inode_names.index((tail_comp_name, tail_state_name))
                if tail_idx not in G.nodes():
                    i_node_name = '{}\n{}'.format(_process_node_name(tail_comp_name),
                                                  _process_node_name(tail_state_name))
                    G.add_node(tail_idx, node_type='i', label=i_node_name)
                G.add_edge(tail_idx, _snode_graph_idx)

            # Iterate S-node indices
            _snode_graph_idx += 1
            _snode_label_idx += 1
        return G
        
    def draw(
            self,
            inference=None,
            layout=None,
            show=True,
            save_file=None,
            dpi=None,
            size_multiplier=None,
            ):
        """ The main function to draw the BKB or a BKB inference.
        
            :param inference: A complete inference returned by reasoning over the bkb.
            :type list:
            :param layout: The type of NetworkX Digraph layout to use.
            :type layout: str
            :param show: Whether to show the graph.
            :type show: bool
            :param save_file: A file path to save the associated figure.
            :type save_file: str
            :param dpi: The DPI to use when graphing the figure.
            :type dpi: int
            :param size_multiplier: A coefficient to multiple the default image size.
            :type size_multiplier: float

            :returns: The matplotlib figure of the associated graph.
            :rtype: matplotlib.figure
        """
        if inference is None:
            G = self._build_bkb_graph()
        else:
            G = self._build_inference_graph(inference)

        #-- Draw s_nodes and i_nodes
        i_nodes = self.bkb.getINodeNames()


        fig, ax = plt.subplots()

        #-- Detect Cycles
        try:
            cycles = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            cycles = None
        if cycles is None and layout != 'neato':
            pos = nx.drawing.nx_agraph.graphviz_layout(G, 'dot')
        else:
            pos = nx.drawing.nx_agraph.graphviz_layout(G, 'neato')

        #-- Collect Inode and Snode list
        inode_list = list()
        snode_list = list()
        for idx, data in G.nodes(data=True):
            if data['node_type'] == 'i':
                inode_list.append(idx)
            elif data['node_type'] == 's':
                snode_list.append(idx)

        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=inode_list, node_shape='s', node_size=500, alpha=.1)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=snode_list, node_shape='o', node_size=100, node_color=(1,1,1), alpha=.1)
        nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', arrowsize=20, alpha=.1, min_source_margin=10, min_target_margin=10)
        nx.draw_networkx_labels(G, pos, {idx: G.nodes[idx]['label'] for idx in list(G.nodes())}, ax=ax)

        ax.set_picker(50)

        ax.set_title(self.bkb.getName())

        plt.tight_layout()
        # Add probability to graph if inference graph
        if inference is not None:
            plt.text(.5, .5, "Probability = {}".format(inference["prob"]),
                ha="center",
                va="center",
                )
        default_size = fig.get_size_inches()
        if size_multiplier is not None:
            new_size = (default_size[0]*size_multiplier, default_size[1]*size_multiplier)
            fig.set_size_inches(*new_size)
        if save_file is not None:
            plt.savefig(save_file, dpi=dpi)
        if show:
            plt.show()
        return fig

if __name__ == '__main__':
    bkb = BKB(name='Aquatic Ecosystem BKB Graph')
    bkb.load('../../examples/aquatic_eco.bkb')

    graph = BkbGraph(bkb)
    graph.show()











