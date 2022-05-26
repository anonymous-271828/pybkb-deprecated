"""
Primary BKB classes.
"""
import json
import uuid
import compress_pickle
import itertools
import math
import tqdm
import numpy as np
from scipy.sparse import dok_array, vstack, hstack
from collections import defaultdict

from pybkb.exceptions import *
from pybkb.legacy.bayesianKnowledgeBase import bayesianKnowledgeBase as legacyBKB
from pybkb.scores import *
from pybkb.utils.probability import build_probability_store

class BKB:
    def __init__(
            self,
            name:str=None,
            sparse_array_format:str='dok',
            ) -> None:
        """ The Bayesian Knowledge Base class.

        Kwargs:
            :param name: Name of the BKB. If None, will assign a random ID to the BKB.
            :type name: str

        Attributes:
            :param adj: The BKB adjacency matrix. Upon BKB build completion, it is a 3-dimensional array
                such that dims 0 and 1 form a N x N matrix where N is number of I-nodes in the BKB, and 
                each 2-d plane (2-dim) is a BKB S-node adjacency matrix where a single non-zero element 
                on the diagonol represents the head of the S-node and all other non-zero elements form the
                S-node tail. Each S-node matrix is symmetric for added utilty later.
            :type adj: numpy.ndarray
            :param inodes: An N x 2 array representing all the I-nodes in the BKB.
            :type inodes: numpy.ndarray
            :param inodes_map: A dictionary with components as keys and associated component states as values.
            :type inodes_map: dict
            :param inodes_indices_map: A map from I-node tuple space to index in the inodes attribute. Used for speed.
            :type inodes_indices_map: dict
            :param snode_probs: A list of S-node probabilities matching the 2-d index of the adjacency matrix.
            :type snode_probs: list
        """
        if name is None:
            name = str(uuid.uuid4())
        self.name = name
        self.head_adj = None
        self.tail_adj = None
        self.inodes = None
        self.inodes_map = defaultdict(list)
        self.inodes_indices_map = {}
        self.snode_probs = None
        self._snodes = None
        if sparse_array_format == 'dok':
            self.sparse_array_obj = dok_array
        elif sparse_array_format == 'csr':
            raise NotImplementedError('Have not implemented CSR matrices, use dok.')
        elif sparse_array_format == 'lil':
            raise NotImplementedError('Have not implemented LiL matrices, use dok.')
        else:
            raise ValueError(f'Unknown sparse array format: {sparse_array_format}.')

    def re_annotate_features(self, annotation_map, append=True):
        """ Will append or replace RV names with new names based on annotation_map.
        """
        for old_rv_name, new_rv_name in annotation_map.items():
            # Gather all inode indices that have feature equalling old name
            inode_indices_to_update = []
            for state in self.inodes_map[old_rv_name]:
                inode_indices_to_update.append(self.inodes_indices_map[(old_rv_name, state)])
            # Create new feature name
            if append:
                new_name = f'{old_rv_name}-{new_rv_name}'
            else:
                new_name = new_rv_name
            # Update name in inodes states map
            self.inodes_map[new_name] = self.inodes_map[old_rv_name]
            # Remove the old name
            self.inodes_map.pop(old_rv_name)
            # Update each index
            for inode_idx in inode_indices_to_update:
                _, state = self.inodes[inode_idx]
                new_inode = (new_name, state)
                # Update name in inodes list
                self.inodes[inode_idx] = new_inode
                # Update the inodes indices map
                self.inodes_indices_map[new_inode] = inode_idx

    def save(self, filepath:str, compression:str='lz4') -> None:
        """ Method to save BKB via compress pickle.

        Args:
            :param filepath: Filepath to save BKB.
            :type filepath: str
        
        Kwargs:
            :param compression: Type of compression compress pickle should use.
                See compress pickle documentation for options.
            :type compressoin: str
        """
        bkb_obj = (
                self.name,
                self.head_adj,
                self.tail_adj,
                self.inodes,
                self.snode_probs,
                )
        with open(filepath, 'wb') as bkb_file:
            compress_pickle.dump(bkb_obj, bkb_file, compression=compression)
        return

    def dumps(self, to_numpy:bool=False):
        """ Dumps the BKB in a tuple of associated numpy arrays.
        """
        if to_numpy:
            return self.name, self.head_adj.toarray(), self.tail_adj.toarray(), self.inodes, np.array(self.snode_probs)
        return self.name, self.head_adj, self.tail_adj, self.inodes, self.snode_probs

    def _load_snodes(self):
        """ Loads S-nodes into an efficient structure. Has a performance hit due
            to looking up indices in the adjacency matrices.
        """
        self._snodes = []
        for snode_index, snode_prob in enumerate(self.snode_probs):
            # Get head I-node from top right outgoing S-node adj (np.array(rows_coords), np.array(cols_coords))
            head_idx = self.head_adj[:,[snode_index]].nonzero()[0][0]
            # Get tail indices from bottom left incoming S-node adj
            tail_indices = self.tail_adj[[snode_index], :].nonzero()[1]
            self._snodes.append((head_idx, snode_prob, tail_indices))

    @classmethod
    def loads(cls, bkb_obj, sparse_array_format:str='dok'):
        bkb = cls(bkb_obj[0], sparse_array_format)
        bkb.head_adj = bkb.sparse_array_obj(bkb_obj[1])
        bkb.tail_adj = bkb.sparse_array_obj(bkb_obj[2])
        bkb.inodes = bkb_obj[3]
        bkb.snode_probs = list(bkb_obj[4])
        # Construct inode maps
        inodes_map = defaultdict(list)
        inodes_indices_map = {}
        for idx, inode in enumerate(bkb.inodes):
           inodes_map[inode[0]].append(inode[1])
           # Gonna be the reverse order
           inodes_indices_map[tuple(inode)] = idx
        bkb.inodes_map = inodes_map
        bkb.inodes_indices_map = inodes_indices_map
        # Load efficient snode internal structure
        bkb._load_snodes()
        return bkb

    @classmethod
    def load_legacy(cls, filepath:str, use_pickle:bool=True, compress:bool=True, sparse_array_format:str='dok'):
        """ Legacy loader of legacy BKB objects. Will soon be deprecated.
        """
        # Load the legacy BKB object
        if use_pickle:
            legacy = legacyBKB().load(filepath, use_pickle, compress)
        else:
            legacy = legacyBKB()
            legacy.load(filepath, use_pickle, compress)
        # Convert to new representation
        bkb = cls(sparse_array_format=sparse_array_format)
        # Add in all I-nodes
        for feature, state in legacy.getINodeNames():
            bkb.add_inode(feature, state)
        # Add in all S-nodes
        for snode in legacy.getAllSNodes():
            head_feature_idx, head_state_idx = snode.getHead()
            head_feature = legacy.getComponentName(head_feature_idx)
            head_state = legacy.getComponentINodeName(head_feature_idx, head_state_idx)
            prob = snode.getProbability()
            tail = []
            for tail_idx in range(snode.getNumberTail()):
                tail_feature_idx, tail_state_idx = snode.getTail(tail_idx)
                tail_feature = legacy.getComponentName(tail_feature_idx)
                tail_state = legacy.getComponentINodeName(tail_feature_idx, tail_state_idx)
                tail.append((tail_feature, tail_state))
            bkb.add_snode(head_feature, head_state, prob, tail)
        return bkb

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
        with open(filepath, 'rb') as bkb_file:
            return cls.loads(compress_pickle.load(bkb_file, compression=compression), sparse_array_format)

    def get_snode_head(self, snode_index:int) -> tuple:
        # Get head I-node from top right outgoing S-node adj (np.array(rows_coords), np.array(cols_coords))
        #head_idx = self.head_adj[:,[snode_index]].nonzero()[0][0]
        head_idx = self._snodes[snode_index][0]
        return tuple(self.inodes[head_idx])

    def get_snode_tail(self, snode_index:int) -> list:
        tail = []
        # Get tail indices from bottom left incoming S-node adj
        #tail_indices = self.tail_adj[[snode_index], :].nonzero()[1]
        tail_indices = self._snodes[snode_index][2]
        # Get Tail I-nodes
        if tail_indices is not None:
            for tail_idx in tail_indices:
                tail.append(tuple(self.inodes[tail_idx]))
        return tail

    def score(
            self,
            data:np.array,
            feature_states:list,
            score_name:str,
            feature_states_index_map:dict=None,
            only:str=None,
            store:dict=None,
            is_learned:bool=False,
            ):
        """ Will score the BKB based on the passed score name.
        
        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param score: The name of the score to use: [mdl_ent, mdl_mi].
            :type str:

        Kwargs:
            :param feature_states_index_map: A dictionary mapping feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map: dict
            :param only: Return only the data score or model score or both. Options: data, model, both, None. Defaults to None which means both.
            :type only: str
            :param store: A store database of calculated joint probabilties.
            :type store: dict
            :param is_learned: If this is a learned BKB from data then we can use a different MDL calculation
                since each learned data instance BKF corresponds to an inference, i.e. node encoding length is just
                log_2(num_features) instead of log_2(num_inodes) as each BKF will have at most one instantiation of a feature.
            :type is_learned: bool

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
        # Build all score nodes from each S-node
        score_nodes = []
        if is_learned:
            node_encoding_len = np.log2(len(self.non_source_features))
        else:
            node_encoding_len = np.log2(len(self.non_source_inodes))
        # Collect score nodes and S-nodes by inferences
        snodes_by_inferences = defaultdict(list)
        for snode_idx in range(len(self.snode_probs)):
            head = self.get_snode_head(snode_idx)
            if '__Source__' in head[0]:
                continue
            tail = self.get_snode_tail(snode_idx)
            # Capture the number of sources supporting the S-node
            if is_learned:
                src_inode = [tail_inode for tail_inode in tail if '__Source__' in tail_inode[0]][0]
                src_feature, src_collection = src_inode
                snodes_by_inferences[snode_idx].extend(list(src_collection))
            # Or if it is not a learned bkb or just an inference bkb
            else:
                snodes_by_inferences[snode_idx].append(None)
            # Remove source nodes from tail
            tail = [tail_inode for tail_inode in tail if '__Source__' not in tail_inode[0]]
            score_nodes.append(
                    score_node_obj(head, node_encoding_len, pa_set=tail, indices=False)
                    )
        # Calculate Scores
        model_score = 0
        data_score = 0
        for snode_idx, node in enumerate(score_nodes):
            _dscore, _mscore, _ = node.calculate_score(
                    data,
                    feature_states,
                    store,
                    feature_states_index_map,
                    only='both',
                    )
            model_score += _mscore
            # Multiply data score by number of learned inferences contain the snodes
            data_score += (_dscore * len(snodes_by_inferences[snode_idx]))
        if only is None:
            return model_score + data_score
        if only == 'data':
            return data_score
        if only == 'model':
            return model_score
        if only == 'both':
            return data_score, model_score
        raise ValueError('Unknown value for only')

    def _snode_to_dict(self, snode_idx:int, prob:float, make_json_serializable:bool=True) -> dict:
        """ Internal method used to form a S-node dictionary representation.
        """
        snode_dict = {}
        head_comp, head_state = self.get_snode_head(snode_idx)
        if make_json_serializable and type(head_state) == frozenset:
            head_state = str(set(head_state))
        snode_dict["Head"] = {head_comp: head_state}
        snode_dict["Probability"] = prob
        tail = self.get_snode_tail(snode_idx)
        # Convert tail list to  dict
        tail_dict = {}
        for feature, state in tail:
            if make_json_serializable and type(state) == frozenset:
                state = str(set(state))
            tail_dict[feature] = state
        snode_dict["Tail"] = tail_dict
        return snode_dict

    def _snodes_to_list(self, make_json_serializable:bool=True) -> list:
        """ Internal method used to gather S-node dictionary representations.
        """
        snodes_list = []
        for idx, prob in enumerate(self.snode_probs):
            snodes_list.append(self._snode_to_dict(idx, prob, make_json_serializable))
        return snodes_list
                        
    def to_dict(self, make_json_serializable:bool=True) -> dict:
        """ Method that transformers BKB to dictionary object.
        """
        snodes_list = self._snodes_to_list(make_json_serializable)
        if make_json_serializable:
            inodes = defaultdict(list)
            for feature, states in self.inodes_map.items():
                for state in states:
                    if type(state) == frozenset:
                        state = str(set(state))
                    inodes[feature].append(state)
        else:
            inodes = dict(self.inodes_map)
        return {
                "Name": self.name,
                "Instatiation Nodes": inodes,
                "Support Nodes": snodes_list,
                }

    def json(self, indent:int=2) -> str:
        """ Builds a JSON BKB object.

        Kwargs:
            :param indent: Amount of indent to use in the json formatting.
            :type indent: int
        """
        bkb_dict = self.to_dict()
        return json.dumps(bkb_dict, indent=indent)
    
    @property
    def source_features(self):
        src_features = []
        for feature in self.inodes_map:
            if '__Source__' in feature:
                src_features.append(feature)
        return src_features
    
    @property
    def source_inodes(self):
        src_inodes = []
        for feature, states in self.inodes_map.items():
            if '__Source__' in feature:
                for state in states:
                    src_inodes.append((feature, state))
        return src_inodes

    @property
    def non_source_features(self):
        nonsrc_features = []
        for feature in self.inodes_map:
            if '__Source__' in feature:
                continue
            nonsrc_features.append(feature)
        return nonsrc_features

    @property
    def non_source_inodes(self):
        nonsrc_inodes = []
        for feature, states in self.inodes_map.items():
            if '__Source__' in feature:
                continue
            for state in states:
                nonsrc_inodes.append((feature, state))
        return nonsrc_inodes
            
    def _add_inode_to_adj(self) -> None:
        """ Internal method that adds I-node rows to the BKBs adjacency matrix.
        """
        if self.head_adj is not None and self.tail_adj is not None:
            # Pad to both head and tail adjacencies
            # Append zeros row to head adjacency (hacked by just changing the shape of the sparse matrix)
            self.head_adj._shape = (self.head_adj.shape[0] + 1, self.head_adj.shape[1])
            # Append zeros column to tail adjacency (hacked by just changing the shape of the sparse matrix)
            self.tail_adj._shape = (self.tail_adj.shape[0], self.tail_adj.shape[1] + 1)
        return

    def add_inode(self, component_name:str, state_name) -> None:
        """ Function to add I-node to BKB.

        Args:
            :param component_name: Name of the I-nodes component (feature) that should be added.
            :type component_name: str, int
            :param state_name: Name of the I-node's state (feature state) that should be added.
            :type state_name: str, int, frozenset
        """
        # Define I-node tuple
        inode_tup = (component_name, state_name)
        if self.inodes_map:
            # Check if inode already exists and raise error if it is.
            comp_states = self.inodes_map.get(component_name, None)
            if comp_states:
                if state_name in comp_states:
                    return
        if self.inodes is None:
            # Create inodes list  and add in inode
            self.inodes = [inode_tup]
        else:
            self.inodes.append(inode_tup)
        # Add to inode maps
        self.inodes_map[component_name].append(state_name)
        self.inodes_indices_map[inode_tup] = len(self.inodes) - 1
        # Update adjacency matrix
        self._add_inode_to_adj()
        return

    def _get_inode_index(self, component_name:str, state_name) -> int:
        """ Helper function to retrieve true I-node index as it is in the adjacency matrix.

        Remember that I-nodes are stacked on top of the adj matrix and S-nodes are appended to the bottom,
        and the inodes_indices_map is the insertion index, not the actual index in the adj. 
        Therefore, this method converts the index to the true index.
        """
        map_idx = self.inodes_indices_map.get((component_name, state_name), None)
        if map_idx is None:
            raise NoINodeError(component_name, state_name)
        return map_idx

    def _check_snode(self,
            target_componet_name:str,
            target_state_name,
            prob:float,
            tail:list,
            ignore_prob:bool=False,
            ) -> True:
        """ Interal method to check S-node for validity and to translate i-node names into indices.
        """
        if not ignore_prob:
            if prob > 1 or prob < 0:
                raise SNodeProbError(prob)
        head_idx = self._get_inode_index(target_componet_name, target_state_name)
        if tail:
            tail_indices = []
            for comp_name, state_name in tail:
                tail_idx = self._get_inode_index(comp_name, state_name)
                if tail_idx is None:
                    raise NoINodeError(comp_name, state_name)
                tail_indices.append(tail_idx)
        else:
            tail_indices = None
        return head_idx, prob, tail_indices

    def add_snode(
            self,
            target_componet_name:str,
            target_state_name,
            prob:float,
            tail:list=None,
            ignore_prob:bool=False,
            ) -> None:
        """ Method to add a Support Node (S-node) to the BKB.

        Args:
            :param target_componet_name: Name of the S-node's head I-node's component (feature).
            :type target_componet_name: str, int
            :param target_state_name: Name of the S-node's head I-node's component state (feature state).
            :type target_state_name: str, int, frozenset
            :param prob: The probability associated with the S-node.
            :type prob: float

        Kwargs:
            :param tail: An optional list of the S-nodes tail I-nodes formed as a list of tuples
                like (component_name, state_name).
            :type tail: list
            :param ignore_prob: Helper flag that allows S-node to have prob values less than 0 or greater than 1.
                Use with caution.
            :type ignore_prob: bool
        """
        # Check S-node validity
        head_idx, prob, tail_indices = self._check_snode(
                target_componet_name,
                target_state_name,
                prob,
                tail,
                ignore_prob,
                )
        # Pad adjanency from the bottom
        if self.head_adj is None and self.tail_adj is None:
            self.head_adj = self.sparse_array_obj(np.zeros((len(self.inodes), 1), dtype=int))
            self.tail_adj = self.sparse_array_obj(np.zeros((1, len(self.inodes)), dtype=int))
        else:
            self.head_adj._shape = (self.head_adj.shape[0], self.head_adj.shape[1] + 1)
            self.tail_adj._shape = (self.tail_adj.shape[0] + 1, self.tail_adj.shape[1])
        # Now add the incoming tails to the S-node
        if tail_indices:
            for tail_idx in tail_indices:
                self.tail_adj[-1,tail_idx] = 1
        # Now add the outgoing head of the S-node
        self.head_adj[head_idx,-1] = 1
        # If None overwride
        if self.snode_probs is None:
            self.snode_probs = [prob]
            self._snodes = [(head_idx, prob, tail_indices)]
        else:
            self.snode_probs.append(prob)
            self._snodes.append((head_idx, prob, tail_indices))
        # Return the S-node index
        return len(self.snode_probs) - 1

    def find_snodes(self, target_componet_name:str, target_state_name, prob:float=None, tail_subset:list=None):
        head_idx = self._get_inode_index(target_componet_name, target_state_name)
        # Find all S-nodes with this head 
        # Will return a tuple of the form (array(zeros), array(snode column indices))
        snode_indices = self.head_adj[[head_idx],:].nonzero()[1]
        # Match snodes with matched criteria
        if prob is None and tail_subset is None:
            return snode_indices
        elif prob and tail_subset is None:
            return [snode_idx for snode_idx in snode_indices if self.snode_probs[snode_idx] == prob]
        elif prob is None and tail_subset:
            _snode_indices = []
            for snode_idx in snode_indices:
                tail = self.get_snode_tail(snode_idx)
                if len(set.intersection(*[set(tail_subset), set(tail)])) == len(tail_subset):
                    _snode_indices.append(snode_idx)
            return _snode_indices
        _snode_indices = []
        for snode_idx in snode_indices:
            if self.snode_probs[snode_idx] != prob:
                continue
            tail = self.get_snode_tail(snode_idx)
            if len(set.intersection(*[set(tail_subset), set(tail)])) == len(tail_subset):
                _snode_indices.append(snode_idx)
        return _snode_indices

    @property
    def snodes_by_head(self):
        snodes_by_head = defaultdict(list)
        for snode_idx in range(len(self.snode_probs)):
            snodes_by_head[self.get_snode_head(snode_idx)].append(snode_idx)
        return snodes_by_head 
   
    @property
    def snodes_by_tail(self):
        snodes_by_tail = defaultdict(list)
        for snode_idx in range(len(self.snode_probs)):
            snodes_by_tail[frozenset(self.get_snode_tail(snode_idx))].append(snode_idx)
        return snodes_by_tail

    def are_snodes_mutex(self, snode_index1, snode_index2, check_head=True):
        # Check head
        if check_head and self.get_snode_head(snode_index1) != self.get_snode_head(snode_index2):
            return True
        # Check tail
        tail1 = set(self.get_snode_tail(snode_index1))
        tail2 = set(self.get_snode_tail(snode_index2))
        # First case: If both tails are empty
        if len(tail1) == len(tail2) == 0:
            return False
        # Second case: If one tail is empty and the other is not
        if len(tail1) == 0 or len(tail2) == 0:
            return False
        # Third case: At least one instantiation differs
        tail_diff = set.symmetric_difference(*[tail1, tail2])
        if len(tail_diff) == 0:
            return False
        # Organize tail difference by features
        tail_diff_dict = defaultdict(list)
        for tail_feature, tail_state in tail_diff:
            tail_diff_dict[tail_feature].append(tail_state)
        # Go through the difference dict and see if any features have more than one state assigned
        for feature, states in tail_diff_dict.items():
            if len(states) == 2:
                return True
        return False

    def is_mutex(self, verbose=False) -> bool:
        # Find S-nodes that have the same head instantiation
        for adj_idx in tqdm.tqdm(range(len(self.inodes)), desc='Checking Head I-nodes', disable=not verbose):
            test_mutex_snodes = self.head_adj[[adj_idx], :].nonzero()[1] 
            # Test every pair of S-nodes to ensure all are mutex with each other (may be a faster way to do this)
            n = len(test_mutex_snodes)
            try:
                total_pairs_to_check = math.factorial(n) / (math.factorial(n-2)*math.factorial(2))
            except ValueError:
                total_pairs_to_check = 0
            if verbose:
                tqdm.tqdm.write(f'Checking {total_pairs_to_check} pairs of S-nodes')
            for snode_idx1, snode_idx2 in itertools.combinations(test_mutex_snodes, r=2):
                # Don't check
                if snode_idx1 == snode_idx2:
                    continue
                # Check
                if not self.are_snodes_mutex(snode_idx1, snode_idx2, check_head=False):
                    raise BKBNotMutexError(snode_idx1, snode_idx2)
        return True
    
    def get_causal_ruleset(self) -> dict:
        """ Returns a dictionary keyed by feature name with list values corresponding to
            the S-node index in each causal ruleset.
        """
        crs = defaultdict(list)
        for adj_idx in range(len(self.inodes)):
            feature, state = self.inodes[adj_idx]
            # Will return a tuple of the form (array(zeros), array(snode column indices))
            snode_indices = self.head_adj[[adj_idx],:].nonzero()[1]
            # Subtract number of S-nodes to get true indices
            crs[feature].extend(list(snode_indices))
        return crs

    def __eq__(self, other) -> bool:
        # First check to see if the inodes are the same
        C = set.intersection(*[set(self.inodes), set(other.inodes)])
        if len(C) != len(self.inodes):
            return False
        # Check to see that all S-node probabilities are the same
        if len(np.intersect1d(np.array(self.snode_probs), np.array(other.snode_probs))) != len(set(self.snode_probs)):
            return False
        # If both easy tests pass, check each S-node structure
        for snode_idx1, prob1 in enumerate(self.snode_probs):
            found = False
            for snode_idx2, prob2 in enumerate(other.snode_probs):
                # Check probability
                if prob1 != prob2:
                    continue
                # Check structure
                # Check if head is the same
                if self.get_snode_head(snode_idx1) != other.get_snode_head(snode_idx2):
                    continue
                # Check if tail is the same
                if set(self.get_snode_tail(snode_idx1)) != set(other.get_snode_tail(snode_idx2)):
                    continue
                # Passed all tests, so we found the same S-node, break
                found = True
                break
            # Check if the S-node was found, if not return false
            if not found:
                return False
        return True
    
    def __hash__(self):
        bkb_obj = (
                self.name,
                self.head_adj,
                self.tail_adj,
                self.inodes,
                self.snode_probs,
                )
        return hash(bkb_obj)
