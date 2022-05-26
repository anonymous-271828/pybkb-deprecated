import os
import operator
import random
import itertools
import logging, tqdm
import json, compress_pickle, pickle
from collections import defaultdict
import numpy as np

import pybkb.python_base.generics.sample_graphs as sample_graphs

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

with open(os.path.join(os.path.dirname(sample_graphs.__file__), 'test_polyforest_graphs.pk'), 'rb') as f_:
    TEST_GRAPHS = pickle.load(f_)

with open(os.path.join(os.path.dirname(sample_graphs.__file__), 'test_polyforest_results.pk'), 'rb') as f_:
    TEST_POLYFORESTS = pickle.load(f_)

class GenericGraph:
    def __init__(
            self,
            adj,
            vertices=None,
            ) -> None:
        """ Expects a numpy adjacency matrix of the form:
            adj = np.array([
                [weight_0-0, weight_0-1, ..., weight_0-N],
                ...
                [weight_N-0, weight_N-1, ..., weight_N-N],
                ])
            where the adj has dimensions NxN such that N = len(vertices).as_integer_ratio
        
        Args:
            :param adj: A numpy NxN numpy array representing the adjacency matrix of the graph. A zero weight means no edge exists.
            :type adj: numpy.array
            :param vertices: A list of vertices of the graph.
            :type vertices: list
        
        Testing:
        >>> import copy
        >>> test_graphs = [GenericGraph(adj, v) for adj, v in TEST_GRAPHS]
        >>> polyforest_results = [GenericGraph(adj, v) for adj, v in TEST_POLYFORESTS]
        >>> for G, P in zip(copy.deepcopy(test_graphs), copy.deepcopy(polyforest_results)):
        ...     p = G.get_polyforest(ensure_no_bidirection=False)
        ...     assert P == p
        >>> for G, P in zip(copy.deepcopy(test_graphs), copy.deepcopy(polyforest_results)):
        ...     p = G.get_polyforest(ensure_no_bidirection=True)
        """
        if vertices is None:
            vertices = [i for i in range(adj.shape[0])]
        self.vertices = vertices
        self.adj = adj

    def get_formatted_adj(self, as_json=False):
        formatted_adj = defaultdict(dict)
        for i in range(len(self.vertices)):
            for j in range(len(self.vertices)):
                if self.adj[i,j] != 0:
                    if as_json:
                        formatted_adj[str(self.vertices[i])][str(self.vertices[j])] = float(self.adj[i,j])
                    else:
                        formatted_adj[self.vertices[i]][self.vertices[j]] = self.adj[i,j]
        return dict(formatted_adj)
    

    def add_edge(
            self,
            weight,
            source_vertex:str=None,
            target_vertex:str=None,
            source_idx:str=None,
            target_idx:str=None,
            ) -> None:
        if source_vertex and source_idx:
            raise ValueError('Specifiy either source vertex or idx, not both.')
        if target_vertex and target_idx:
            raise ValueError('Specifiy either target vertex or idx, not both.')
        if source_vertex:
            source_idx = self.vertices.index(source_vertex)
        if target_vertex:
            target_idx = self.vertices.index(target_vertex)
        self.adj[source_idx][target_idx] = weight

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent:list, rank:list, x, y) -> None:
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def remove_bidirectional_edges(self, eval_fn:str='max'):
        """ Based on the eval_fn if two edges exist between two nodes (i.e. bidirectional) it
        will take the edge that is best given the eval_fn.
        """
        if eval_fn == 'min':
            eval_fn = min
        elif eval_fn == 'max':
            eval_fn = max
        else:
            raise ValueError(f'Unknown eval_fn: {eval_fn}')
        result_adj = np.zeros_like(self.adj)
        processed_edges = set()
        for i, j, weight_ij in self:
            if (i,j) in processed_edges:
                continue
            # Check opposite direction
            weight_ji = self.adj[j,i]
            # If there is already no edge continue
            if weight_ij == weight_ji == 0:
                continue
            elif weight_ij == weight_ji:
                # Choose a random edge
                if bool(random.getrandbits(1)):
                    result_adj[i,j] = weight_ij
                else:
                    result_adj[j,i] = weight_ji
            elif eval_fn(weight_ij, weight_ji) == weight_ij:
                result_adj[i,j] = weight_ij
            elif eval_fn(weight_ij, weight_ji) == weight_ji:
                result_adj[j,i] = weight_ji
            else:
                raise RuntimeError(f'Not sure which edge to take at indices ({i}, {j}).')
            processed_edges.add((i,j))
            processed_edges.add((j,i))
        return GenericGraph(
                result_adj,
                vertices=self.vertices,
                )

    def get_disconnected_nodes(self):
        disconntect_sum_list = []
        abs_adj = np.abs(self.adj)
        for axis in range(self.adj.ndim):
            disconntect_sum_list.append(
                    np.sum(abs_adj, axis=axis)
                    )
        disconnected_indices = np.where(np.sum(disconntect_sum_list) == 0)[0]
        return disconnected_indices

    
    def get_polyforest(self, eval_fn:str='max', ensure_no_bidirection:bool=True):
        """ Given a weighted adjecency matrix G, will return a polyforest based on passed
            criteria.

            :param eval_fn: A function that will evalute weights of candidate edges and return the "best" edge. Default: max.
            :type eval_fn: function

            :return: Polyforest.
            :rtype: GenericGraph

        """
        if ensure_no_bidirection:
            G = self.remove_bidirectional_edges(eval_fn=eval_fn)
        else:
            G = self
        result_adj = np.zeros_like(G.adj)
        i, e = 0, 0
        if eval_fn == 'max':
            reverse = True
        elif eval_fn == 'min':
            reverse = False
        else:
            ValueError('Unknown eval function. Please choose either min or max.')
        edges = sorted(G, key=lambda item: item[2], reverse=reverse)
        disconnected_nodes = G.get_disconnected_nodes()
        parent = []
        rank = []
        for node in range(len(G.vertices)):
            parent.append(node)
            rank.append(0)
        while e < len(G.vertices) - 1 - disconnected_nodes.size:
            try:
                source, target, weight = edges[i]
            except IndexError:
                break
            i += 1
            x = G.find(parent, source)
            y = G.find(parent, target)
            if x != y:
                e += 1
                result_adj[source, target] = weight
                G.apply_union(parent, rank, x, y)
        return GenericGraph(
                result_adj,
                vertices=self.vertices,
                )

    def thresold(self, cut_off_weight, op='>', verbose=False):
        """ Returns new graph that is thresholded.
            
            :param cut_off_weight: The weight that will be used to threshold all edges.
            :type cut_off_weight: float
            :param weight_idx: The index of the weight for any given edge. Defaults to -1.
            :type weight_idx: int
            :param op: Operation to use: > or <.
            :type op: str
        """
        # Get operator function handle
        if op == '>':
            op = operator.gt
        elif op == '<':
            op = operator.lt
        else:
            raise ValueError('f{op} is unsupported.')
        # Build thresholded adjacency matrix
        thresold_adj = np.zeros_like(self.adj)
        for i, j, weight in tqdm.tqdm(self, desc='Thresholding edges', leave=False, disable=not verbose):
            if op(weight, cut_off_weight):
                thresold_adj[i,j] = weight
        return GenericGraph(
                thresold_adj,
                vertices=self.vertices,
                )

    def save(self, filepath, compress=True, json=False):
        """ Saves as a (compressed) pickle file.
        """
        if json:
            with open(filepath, 'w') as f_:
                json.dump(self.get_formatted_adj(as_json=True), f_)
                return
        if compress:
            with open(filepath, 'wb') as f_:
                compress_pickle.dump((self.adj, self.vertices), f_, 'lz4')
        else:
            with open(filepath, 'wb') as f_:
                pickle.dump((self.adj, self.vertices), f_)
        return 
    
    def subgraph(self, filter_node_indices=None, filter_node_names=None, verbose=False):
        """ Builds a subgraph based on a node filter list.

            :param filter_nodes: A list of nodes to subgraph.
            :type filter_nodes: list
            :param verbose: Enable or Disable tqdm output.
            :type verbose: bool
        """
        if filter_node_indices is not None and filter_node_names is not None:
            raise ValueError('Can not pass both filter node indices and names, choose one option.')
        if filter_node_names is not None:
            filter_node_indices = [self.vertices.index(node_name) for node_name in filter_node_names]
        sub_adj = np.zeros_like(self.adj)
        for perm in tqdm.tqdm(itertools.permutations(filter_node_indices, r=self.adj.ndim), desc='Processing Filter Nodes', disable=not verbose, leave=False):
            sub_adj[perm] = self.adj[perm]
        return GenericGraph(
                sub_adj,
                vertices=self.vertices,
                )

    def edges(self):
        for i, j, weight in self:
            yield i, j, self.vertices[i], self.vertices[j], weight

    def num_edges(self):
        return np.where(self.adj != 0)[0].size

    @classmethod
    def load(cls, filepath, compressed=True, compression='lz4'):
        if compressed:
            with open(filepath, 'rb') as f_:
                adj, vertices =  compress_pickle.load(f_, compression)
        else:
            with open(filepath, 'r') as f_:
                adj, vertices = json.load(f_)
        return cls(adj, vertices=vertices)

    def __len__(self):
        return self.adj.size

    def __iter__(self):
        for i, j in zip(*self.adj.nonzero()):
            yield i, j, self.adj[i,j]

    def __eq__(self, other):
        if other.vertices != self.vertices:
            return False
        return np.array_equal(other.adj, self.adj)

    def __str__(self):
        return json.dumps(self.get_formatted_adj(as_json=True), indent=2)

def test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    test()
