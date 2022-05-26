import json
import uuid
import compress_pickle
import itertools
import math
import tqdm
import numpy as np
import re
from scipy.sparse import dok_array, vstack, hstack
from collections import defaultdict

from pybkb.scores import *
from pybkb.utils.probability import *
from pybkb.bkb import BKB


class BN:
    def __init__(
            self,
            name:str=None,
            sparse_array_format:str='dok',
            ) -> None:
        """ The Bayesian Network utility class. Doesn't have many methods and is just used
            to store general structure and calculate basic quantities.

        Kwargs:
            :param name: Name of the BN. If None, will assign a random ID to the BN.
            :type name: str

        Attributes:
            :param adj: The BN adjacency matrix.
            :type adj: scipy.sparse.dok_array
            :param cpt: Conditional Probability Tables of the BN.
            :type cpt: dict
        """
        if name is None:
            name = str(uuid.uuid4())
        self.name = name
        self.adj = None
        self.rvs = None
        self.rv_map = defaultdict(list)
        self.rv_indices_map = {}
        self.rv_state_indices_map = defaultdict(dict)
        self.cpt = None
        self.pa = None
        if sparse_array_format == 'dok':
            self.sparse_array_obj = dok_array
        elif sparse_array_format == 'csr':
            raise NotImplementedError('Have not implemented CSR matrices, use dok.')
        elif sparse_array_format == 'lil':
            raise NotImplementedError('Have not implemented LiL matrices, use dok.')
        else:
            raise ValueError(f'Unknown sparse array format: {sparse_array_format}.')

    def save(self, filepath:str, compression:str='lz4') -> None:
        """ Method to save BN via compress pickle.

        Args:
            :param filepath: Filepath to save BN.
            :type filepath: str
        
        Kwargs:
            :param compression: Type of compression compress pickle should use.
                See compress pickle documentation for options.
            :type compressoin: str
        """
        bn_obj = (
                self.name,
                self.adj,
                self.cpt,
                self.rvs,
                self.rv_map,
                )
        with open(filepath, 'wb') as bn_file:
            compress_pickle.dump(bn_obj, bn_file, compression=compression)
        return

    def dumps(self, to_numpy:bool=False):
        """ Dumps the BN in a tuple of associated numpy arrays.
        """
        if to_numpy:
            return self.name, self.adj.toarray(), self.cpt, self.rvs, self.rv_map 
        return self.name, self.adj, self.cpt, self.rvs, self.rv_map

    def calculate_pa_from_adj(self):
        self.pa = {}
        for rv_idx, rv in enumerate(self.rvs):
            pa_indices = self.adj[[rv_idx], :].nonzero()[1]
            pa = []
            for pa_idx in pa_indices:
                pa.append(self.rvs[pa_idx])
            if len(pa) == 0:
                self.pa[rv] = None
            else:
                self.pa[rv] = pa
        return

    @classmethod
    def loads(cls, bn_obj, sparse_array_format):
        bn = cls(bn_obj[0], sparse_array_format)
        bn.adj = bn.sparse_array_obj(bn_obj[1])
        # Calculate pa sets from adj
        bn.cpt = bn_obj[2]
        bn.rvs = bn_obj[3]
        bn.rv_map = bn_obj[4]
        bn.calculate_pa_from_adj()
        bn.rv_indices_map = {rv: idx for idx, rv in enumerate(bn.rvs)}
        bn.rv_state_indices_map = defaultdict(dict)
        for rv, states in bn.rv_map.items():
            for state_idx, state in enumerate(states):
                bn.rv_state_indices_map[rv][state] = state_idx
        return bn

    @classmethod
    def load(cls, filepath:str, compression:str='lz4', sparse_array_format:str='dok'):
        """ A class method to load a BKB that was saved via compress_pickle using the BKB.save() method.

        Args:
            :param filepath: Filepath from which to load the BKB compress pickle file.
            :type filepath: str

        Kwargs:
            :param compression: Type of compression that was used to save the BKB.
            :type compression: str
        """
        with open(filepath, 'rb') as bn_file:
            return cls.loads(compress_pickle.load(bn_file, compression=compression), sparse_array_format)

    @classmethod
    def from_bnlearn_modelstr(cls, modelstr, states_map):
        bn = cls()
        for node_info in re.findall('\[.*?\]', modelstr):
            # Remove enclosing brackets
            node_info = re.sub('[\[\]]', '', node_info)
            # Split on parent sets
            try:
                node_name, pa_info = node_info.split('|')
            except ValueError:
                # This means no parent set was found
                bn.add_rv(node_info, states_map[node_info])
                #nodes.append(Node(node_info, states=states_map[node_info]))
                continue
            # Get parent nodes
            pa_names = pa_info.split(':')
            # Add head RV
            bn.add_rv(node_name, states_map[node_name])
            # Add all parent RVs
            for pa_name in pa_names:
                bn.add_rv(pa_name, states_map[pa_name])
            # Add parents to head node
            bn.add_parents(node_name, pa_names)
        return bn

    def add_rv(self, rv_name:str, rv_states:list=None):
        # Add random variable to RV list
        if self.rvs is None:
            self.rvs = [rv_name]
        # Check to make sure rv doesn't already exist
        elif rv_name in self.rvs:
            return
        else:
            self.rvs.append(rv_name)
        # Add to pa
        if self.pa is None:
            self.pa = {rv_name: None}
        else:
            self.pa[rv_name] = None
        # Add to index map
        self.rv_indices_map[rv_name] = len(self.rvs) - 1
        # Add states if passed
        if rv_states is not None:
            self.rv_map[rv_name].extend(rv_states)
        # Add states to state indices map
        for idx, state in enumerate(self.rv_map[rv_name]):
            self.rv_state_indices_map[rv_name][state] = idx
        # Extend adjacency matrix
        if self.adj is None:
            self.adj = self.sparse_array_obj(np.zeros((1, 1), dtype=int))
        else:
            self.adj._shape = (self.adj.shape[0] + 1, self.adj.shape[1] + 1)
        return

    def add_parents(self, rv_name:str, pa_set:list):
        rv_idx = self.rv_indices_map[rv_name]
        pa_indices = [self.rv_indices_map[pa] for pa in pa_set]
        # Add parents to rv pa_set
        if self.pa is None:
            self.pa = {rv_name: pa_set}
        elif self.pa[rv_name] is None:
            self.pa[rv_name] = pa_set
        else:
            self.pa[rv_name].update(set(pa_set))
        # Add to adjacency matrix
        for pa_idx in pa_indices:
            self.adj[rv_idx, pa_idx] = 1
        # Add initialized cpt table
        if self.cpt is None:
            self.cpt = defaultdict(dict)
        for state_idx, _ in enumerate(self.rv_map[rv_name]):
            pa_indices_config_options = [range(len(self.rv_map[pa])) for pa in pa_set]
            for pa_state_config in itertools.product(*pa_indices_config_options):
                pa_config = [(pa_idx, pa_state_idx) for pa_idx, pa_state_idx in zip(pa_indices, pa_state_config)]
                self.cpt[(rv_idx, state_idx)][frozenset(pa_config)] = -1
        return

    def make_bkb(self):
        bkb = BKB(self.name)
        # Add all I-nodes
        for feature, states in self.rv_map.items():
            for state in states:
                bkb.add_inode(feature, state)
        # Add all CPT entries as S-nodes
        for (rv_idx, rv_state_idx), pa_config_probs in self.cpt.items():
            rv = self.rvs[rv_idx]
            rv_state = self.rv_map[rv][rv_state_idx]
            for pa_config, prob in pa_config_probs.items():
                # Form S-node tail
                tail = []
                for pa_idx, pa_state_idx in pa_config:
                    pa = self.rvs[pa_idx]
                    pa_state = self.rv_map[pa][pa_state_idx]
                    tail.append((pa, pa_state))
                # Add S-node
                bkb.add_snode(rv, rv_state, prob, tail=tail)
        return bkb

    def add_cpt_entry(self, rv, rv_state, prob, pa_set_config:list=None):
        rv_idx = self.rv_indices_map[rv]
        rv_state_idx = self.rv_state_indices_map[rv][rv_state]
        if pa_set_config is None:
            pa_indices_config = frozenset()
        else:
            pa_indices_config = []
            for pa_name, pa_state in pa_set_config:
                pa_idx = self.rv_indices_map[pa_name]
                pa_state_idx = self.rv_state_indices_map[pa_name][pa_state]
                pa_indices_config.append((pa_idx, pa_state_idx))
            pa_indices_config = frozenset(pa_indices_config)
        self.cpt[(rv_idx, rv_state_idx)][pa_indices_config] = prob
    
    def calculate_cpts_from_data(self, data, feature_states):
        feature_states_indices_map = {fs: idx for idx, fs in enumerate(feature_states)}
        store = build_probability_store()
        # First calculate all nodes with no parents
        for rv, pa_set in self.pa.items():
            if pa_set is not None:
                continue
            rv_idx = self.rv_indices_map[rv]
            for rv_state_idx, rv_state in enumerate(self.rv_map[rv]):
                x_state_idx = feature_states_indices_map[(rv, rv_state)]
                prob, store = joint_prob(data, x_state_idx, [], store)
                self.cpt[(rv_idx, rv_state_idx)][frozenset()] = prob
        # Calculate the rest
        for rv_state_tup, pa_probs in self.cpt.items():
            rv_idx, rv_state_idx = rv_state_tup
            rv = self.rvs[rv_idx]
            rv_state = self.rv_map[rv][rv_state_idx]
            x_state_idx = feature_states_indices_map[(rv, rv_state)]
            for pa_config, _ in pa_probs.items():
                pa_indices = []
                for pa_idx, pa_state_idx in pa_config:
                    pa = self.rvs[pa_idx]
                    pa_state = self.rv_map[pa][pa_state_idx]
                    pa_indices.append(feature_states_indices_map[(pa, pa_state)])
                p_xp, store = joint_prob(data, x_state_idx, pa_indices, store)
                p_p, store = joint_prob(data, x_state_idx=None, parent_state_indices=pa_indices, store=store)
                try:
                    prob = p_xp / p_p
                except ZeroDivisionError:
                    prob = 0
                self.cpt[rv_state_tup][pa_config] = prob
        return

    def score_like_bkb(
            self,
            data:np.array,
            feature_states:list,
            score_name:str,
            feature_states_index_map:dict=None,
            only:str=None,
            store:dict=None,
            ):
        """ Will score the BN as if it were a learned BKB with corresponding learned fragments
        that are encoded by the BN.
        
        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param score: The name of the score to use: [mdl_ent, mdl_mi].
            :type str:

        Kwargs:
            :param feature_states_map: A dictionary keyed by feature with values of as the list of available states the feature can take. 
                Use the build_feature_state_map function in pybkb.utils.probability to get the correct format.
            :param only: Return only the data score or model score or both. Options: data, model, both, None. Defaults to None which means both.
            :type only: str
            :param store: A store database of calculated joint probabilties.
            :type store: dict
        """
        # Initalize
        if store is None:
            store = build_probability_store()
        if feature_states_index_map is None:
            # Build feature states index map
            feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        # Get score function
        if score_name == 'mdl_ent':
            score_node_obj = MdlEntScoreNode
        elif score_name == 'mdl_mi':
            score_node_obj = MdlMutInfoScoreNode
        else:
            raise ValueError('Unknown score name.')
        node_encoding_len = np.log2(len(self.rvs))
        # Count the unique worlds in the data and extract the instantiated score nodes
        world_counts = defaultdict(int)
        score_nodes_dict = {}
        for data_idx in range(data.shape[0]):
            # Count world
            data_hash = tuple(data[data_idx,:].tolist())
            if data_hash not in world_counts:
                # Extract instantiations
                fs_indices = np.argwhere(data[data_idx,:] == 1).flatten()
                fs_set = [(fs_idx, feature_states[fs_idx]) for fs_idx in fs_indices]
                # Build score nodes
                score_nodes = []
                for fs_idx, fs in fs_set:
                    feature, state = fs
                    # Get parents of this feature
                    pa_feature_set = self.get_parents(feature)
                    if pa_feature_set is None:
                        pa_feature_set = []
                    # Collect parent instantiation set indices
                    pa_fs_set = [ifs[0] for ifs in fs_set if fs[1][0] in pa_feature_set]
                    score_nodes.append(
                            score_node_obj(fs_idx, node_encoding_len, pa_set=pa_fs_set, indices=True)
                            )
                score_nodes_dict[data_hash] = score_nodes
            # Add to world count
            world_counts[data_hash] += 1
        # Calculate Scores
        model_score = 0
        data_score = 0
        for world, score_nodes in score_nodes_dict.items():
            for node in score_nodes:
                _dscore, _mscore, _ = node.calculate_score(
                        data,
                        feature_states,
                        store,
                        feature_states_index_map,
                        only='both',
                        )
                model_score += _mscore
                # Multiply data score by number of learned inferences contain the snodes
                data_score += (_dscore * world_counts[world])
        if only is None:
            return model_score + data_score
        if only == 'data':
            return data_score
        if only == 'model':
            return model_score
        if only == 'both':
            return data_score, model_score
        raise ValueError('Unknown value for only')

    def score(
            self,
            data:np.array,
            feature_states:list,
            score_name:str,
            feature_states_map:dict=None,
            only:str=None,
            store:dict=None,
            ):
        """ Will score the BN based on the passed score name.
        
        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param score: The name of the score to use: [mdl_ent, mdl_mi].
            :type str:

        Kwargs:
            :param feature_states_map: A dictionary keyed by feature with values of as the list of available states the feature can take. 
                Use the build_feature_state_map function in pybkb.utils.probability to get the correct format.
            :param only: Return only the data score or model score or both. Options: data, model, both, None. Defaults to None which means both.
            :type only: str
            :param store: A store database of calculated joint probabilties.
            :type store: dict
        """
        # Initalize
        if store is None:
            store = build_probability_store()
        if feature_states_map is None:
            # Build feature states index map
            feature_states_map = build_feature_state_map(feature_states)
        # Get score function
        if score_name == 'mdl_ent':
            score_node_obj = MdlEntScoreNode
        elif score_name == 'mdl_mi':
            score_node_obj = MdlMutInfoScoreNode
        else:
            raise ValueError('Unknown score name.')
        # Build all score nodes from each S-node
        score_nodes = []
        node_encoding_len = np.log2(len(self.rvs))
        for rv in self.rvs:
            pa_set = self.get_parents(rv)
            if pa_set is None:
                pa_set = []
            score_nodes.append(
                    score_node_obj(
                        rv,
                        node_encoding_len,
                        pa_set=pa_set,
                        indices=False,
                        rv_level=True,
                        states=self.rv_map,
                        )
                    )
        # Calculate Scores
        model_score = 0
        data_score = 0
        for node in score_nodes:
            _dscore, _mscore, _ = node.calculate_score(
                    data,
                    feature_states,
                    store,
                    feature_states_map=feature_states_map,
                    only='both',
                    )
            model_score += _mscore
            data_score += _dscore
        if only is None:
            return model_score + data_score
        if only == 'data':
            return data_score
        if only == 'model':
            return model_score
        if only == 'both':
            return data_score, model_score
        raise ValueError('Unknown value for only')

    def get_parents(self, rv):
        return self.pa[rv]

    def _bn_struct_to_dict(self):
        struct = []
        for rv in self.rvs:
            struct.append(
                    {
                        rv: {
                            "Parents": self.get_parents(rv),
                            }
                        }
                    )
        return struct

    def _format_cpt(self, make_json_serializable):
        cpt = defaultdict(dict)
        for (rv_idx, rv_state_idx), pa_probs in self.cpt.items():
            rv = self.rvs[rv_idx]
            rv_state = self.rv_map[rv][rv_state_idx]
            if make_json_serializable:
                for pa_config, prob in pa_probs.items():
                    pa_config_str = [
                            f'{self.rvs[pa_idx]} = {self.rv_map[self.rvs[pa_idx]][pa_state_idx]}' for pa_idx, pa_state_idx in pa_config
                            ]
                    cpt[f'{rv} = {rv_state}'][', '.join(pa_config_str)] = prob
            else:
                for pa_config, prob in pa_probs.items():
                    pa_config = [
                            (self.rvs[pa_idx], self.rv_map[self.rvs[pa_idx]][pa_state_idx]) for pa_idx, pa_state_idx in pa_config
                            ]
                    cpt[(rv, rv_state)][frozenset(pa_config)] = prob

        return cpt

    def to_dict(self, make_json_serializable:bool=True, include_cpt:bool=False) -> dict:
        """ Method that transformers BN to dictionary object.
        """
        # Make dictionary structure
        bn_struct_dict = self._bn_struct_to_dict()
        # Initialize full dictionary
        d = {
                "Name": self.name,
                "RVs": self.rvs,
                "States": self.rv_map,
                "Structure": bn_struct_dict,
                }
        if include_cpt:
            cpt = self._format_cpt(make_json_serializable) 
            d["CPTs"] = cpt
        return d

    def json(self, indent:int=2, include_cpt:bool=False) -> str:
        """ Builds a JSON BKB object.

        Kwargs:
            :param indent: Amount of indent to use in the json formatting.
            :type indent: int
        """
        bn_dict = self.to_dict(make_json_serializable=True, include_cpt=include_cpt)
        return json.dumps(bn_dict, indent=indent)

    def __eq__(self, other):
        # Check random variable equivalence
        if set(self.rvs) != set(other.rvs):
            return False
        # Check all random variable state equivalence
        for rv in self.rvs:
            if set(self.rv_map[rv]) != set(other.rv_map[rv]):
                return False
            # Check parent structure equivalence
            if self.pa[rv] == self.pa[rv]:
                continue
            elif set(self.pa[rv]) != set(other.pa[rv]):
                return False
        # Check CPTs
        if self.cpt is None and other.cpt is None:
            return True
        # We have already confirmed structure is correct so we can simply extract
        # probs and test that they are exactly the same.
        probs1 = []
        probs2 = []
        for _, pa_probs in self.cpt.items():
            for _, prob in pa_probs.items():
                probs1.append(prob)
        for _, pa_probs in other.cpt.items():
            for _, prob in pa_probs.items():
                probs2.append(prob)
        # Sort them
        probs1 = sorted(probs1)
        probs2 = sorted(probs2)
        if probs1 != probs2:
            return False
        return True
