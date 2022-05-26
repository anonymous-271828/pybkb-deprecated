import numpy as np

from pybkb.utils.probability import *


class ScoreNode:
    def __init__(
            self,
            x,
            node_encoding_len:float,
            rv_level:bool=False,
            pa_set:list=None,
            states:dict=None,
            indices:bool=True,
            )  -> None:
        """ Generic scoring node that captures a parent set and target variable and other generic
            information.

        Args:
            :param x: The target feature or feature state. 
            :type x: str or tuple
            :param node_encoding_len: Encoding length of a node in the model.
            :type node_encoding_len: float

        Kwargs:
            :param rv_level: Whether to score at the random variable level or the random variable instantiation level.
                Defaults to False.
            :type rv_level: bool
            :param pa_set: Parent set of x.
            :type pa_set: list
            :param states: Dictionary of every features set of states. Used in calculating scores on 
            the RV level, not necessary for instantiation level calculations.
            :type states: dict
            :param indices: Specifies if x and pa_set identifiers are indices in the data matrix. Defaults to True.
            :type indices: bool
        """
        if rv_level and indices:
            raise NotImplementedError('Must pass in feature and pa set NAMES not indices.')
        self.rv_level = rv_level
        self.x = x
        if pa_set is None:
            self.pa_set = []
        else:
            self.pa_set = list(pa_set)
        self.node_encoding_len = node_encoding_len
        self.indices = indices
        # Reduces state dictionary to only what is in the node if rv level node calculator
        if states is not None and self.rv_level:
            self.states = {feature: states[feature] for feature in self.pa_set + [x]}
        else:
            self.states = None

    def _calc_instantiated_score(self, data, feature_states_index_map, store):
        pass
    
    def _calc_rvlevel_score(self, data, feature_states, feature_states_map):
        pass

    def calculate_score(
            self,
            data:np.array,
            feature_states:list,
            store:dict,
            feature_states_index_map:dict=None,
            feature_states_map:dict=None,
            only:str=None,
            ):
        """ Generic method that is overwritten to calculate score.
            
        Args:
            :param data: Full database in a binary matrix.
            :type data: np.array
            :param feature_states: A list of all feature states exactly as it appears in the data matrix.
            :type feature_states: list
            :param store: A store database of calculated joint probabilties.
            :type store: dict

        Kwargs:
            :param feature_states_index_map: A dictionary mapping feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map: dict
            :param feature_states_map: A dictionary keyed by feature with values of as the list of available states the feature can take. 
                Use the build_feature_state_map function in pybkb.utils.probability to get the correct format.
            :type feature_states_map: dict
            :param only: Return only the data score or model score or both. Options: data, model, both, None. Defaults to None which means both.
            :type only: str
        """
        if not self.rv_level:
            # Calculate structure MDL
            struct_mdl = (len(self.pa_set) + 1)*self.node_encoding_len
            # Calculate instantiated data MDL
            data_mdl, store = self._calc_instantiated_score(data, feature_states_index_map, store)
            # Note: Number of atomic events represented by an S-node is just 1
        else:
            # Calculate node structure MDL
            struct_mdl = len(self.pa_set) * self.node_encoding_len
            struct_mdl += (len(self.states[self.x]) - 1) * np.prod([len(self.states[pa]) for pa in self.pa_set])
            num_atomic_events = len(self.states[self.x]) * np.prod([len(self.states[pa]) for pa in self.pa_set])
            # Calculate random variable level data MDL
            data_mdl, store = self._calc_rvlevel_score(data, feature_states, feature_states_map, store)
            data_mdl *= num_atomic_events
        if only is None:
            return -data_mdl - struct_mdl, store
        if only == 'data':
            return -data_mdl, store
        if only == 'model':
            return -struct_mdl, store
        if only == 'both':
            return -data_mdl, -struct_mdl, store
        raise ValueError(f'Unknown option {only}. Must be one of [data, model, both].')


class MdlEntScoreNode(ScoreNode):
    def __init__(
            self,
            x,
            node_encoding_len:float,
            rv_level:bool=False,
            pa_set:list=None,
            states:dict=None,
            indices:bool=True,
            )  -> None:
        """ Generic scoring node that captures a parent set and target variable and other generic
            information.

        Args:
            :param x: The target feature or feature state. 
            :type x: str or tuple
            :param node_encoding_len: Encoding length of a node in the model.
            :type node_encoding_len: float

        Kwargs:
            :param rv_level: Whether to score at the random variable level or the random variable instantiation level.
                Defaults to False.
            :param pa_set: Parent set of x.
            :type pa_set: list
            :param states: Dictionary of every features set of states. Used in calculating scores on 
            the RV level, not necessary for instantiation level calculations.
            :type states: dict
            :param indices: Specifies if x and pa_set identifiers are indices in the data matrix. Defaults to True.
            :type indices: bool
        """
        super().__init__(x, node_encoding_len, rv_level, pa_set, states, indices)

    def _calc_instantiated_score(self, data, feature_states_index_map, store):
        """ Calculate MDL with conditional entropy weight on the random variable instantiation level.
        """
        # Get x and pa indices in data set
        if self.indices:
            x_state_idx = self.x
            parent_state_indices = self.pa_set
        else:
            x_state_idx = feature_states_index_map[self.x]
            parent_state_indices = [feature_states_index_map[pa] for pa in self.pa_set]
        # Calculate data MDL
        return instantiated_entropy(data, x_state_idx, parent_state_indices, store)

    def _calc_rvlevel_score(self, data, feature_states, feature_states_map, store):
        """ Calculate MDL with conditional entropy weight on the random variable level.
        """
        # If no parents use variable entropy as data_mdl
        if len(self.pa_set) == 0:
            return entropy(data, self.x, feature_states, feature_states_map, store=store)
        else:
            # Cacluate conditional entropy weight
            return entropy(data, self.x, feature_states, feature_states_map, parents=self.pa_set, store=store)


class MdlMutInfoScoreNode(ScoreNode):
    def __init__(
            self,
            x,
            node_encoding_len:float,
            rv_level:bool=False,
            pa_set:list=None,
            states:dict=None,
            indices:bool=True,
            )  -> None:
        """ Generic scoring node that captures a parent set and target variable and other generic
            information.

        Args:
            :param x: The target feature or feature state. 
            :type x: str or tuple
            :param node_encoding_len: Encoding length of a node in the model.
            :type node_encoding_len: float

        Kwargs:
            :param rv_level: Whether to score at the random variable level or the random variable instantiation level.
                Defaults to False.
            :param pa_set: Parent set of x.
            :type pa_set: list
            :param states: Dictionary of every features set of states. Used in calculating scores on 
            the RV level, not necessary for instantiation level calculations.
            :type states: dict
            :param indices: Specifies if x and pa_set identifiers are indices in the data matrix. Defaults to True.
            :type indices: bool
        """
        super().__init__(x, node_encoding_len, rv_level, pa_set, states, indices)

    def _calc_instantiated_score(self, data, feature_states_index_map, store):
        """ Calculate MDL with mutual information weight on the random variable instantiation level.
        """
        # Get x and pa indices in data set
        if self.indices:
            x_state_idx = self.x
            parent_state_indices = self.pa_set
        else:
            x_state_idx = feature_states_index_map[self.x]
            parent_state_indices = [feature_states_index_map[pa] for pa in self.pa_set]
        # Calculate data MDL
        return instantiated_mutual_info(data, x_state_idx, parent_state_indices, store)

    def _calc_rvlevel_score(self, data, feature_states, feature_states_map, store):
        """ Calculate MDL with mutual information weight on the random variable level.
        """
        if len(self.pa_set) == 0:
            return entropy(data, self.x, feature_states, feature_states_map, store=store)
        else:
            # Cacluate Mutual Information weight
            return mutual_info(data, self.x, self.pa_set, feature_states, feature_states_map, store=store)
