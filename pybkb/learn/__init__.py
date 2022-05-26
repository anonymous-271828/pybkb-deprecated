import numpy as np

from pybkb.learn.backends import *
from pybkb.fusion import fuse


class BKBLearner:
    def __init__(
            self,
            backend:str,
            score:str,
            palim:int=None,
            distributed:bool=False,
            #num_learners:int=None,
            #num_cluster_nodes:int=None,
            ray_address:str=None,
            ) -> None:
        """ The Bayesian Knowledge Base Learner class.

        Args:
            :param backend: Name of the learning backend. Choices: gobnilp or notears.
            :type name: str
            :param score: Name of scoring function. Choices: mdl_mi, mdl_ent.
            :type score: str

        Kwargs:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param distributed: Whether to distrbute learning across a Ray cluster.
            :type distributed: bool
            :param ray_address: The address of the Ray cluster. Defaults to auto.
            :type ray_address: str
        """
        if backend == 'gobnilp':
            if distributed:
                self.backend = BKBGobnilpDistributedBackend(score, palim, ray_address=ray_address)
            else:
                self.backend = BKBGobnilpBackend(score, palim)
        elif backend == 'notears':
            if distributed:
                self.backend = NoTearsDistributedBackend(score)
            else:
                self.backend = NoTearsBackend(score)
        else:
            raise ValueError(f'Unknown backend: {backend}.')

    def fit(
            self,
            data:np.array,
            feature_states:list,
            srcs:list=None,
            src_reliabilities:list=None,
            collapse:bool=True,
            verbose:bool=True,
            ) -> None:
        """ Learn a BKB from data.

        Args:
            :param data: NxM binary matrix where N is the number of data instances and M
            is the number of feature instantiations. Data instance row should have a 1 if 
            the column feature instantiation is present in the instance and zero otherwise.
            :type data: np.array
            :param feature_states: A list of feature instantiation tuples of the form: [(feature1, state1), (feature1, state2),...].
            Must match the M dimension of the data exactly.
            :type feature_states: list

        Kwargs:
            :param srcs: Names of each of the data instances. Should match N dimension of data exactly.
            :type srcs: list
            :param src_reliabilities: Reliabilities of each src in the data instances. Can be any list of numbers, where each
                reliability must be > 0.
            :type src_reliabilities: list
            :param collapse: Will collapse the fused bkb to make a more efficient representation for visualization and reasoning.
            :type collapse: bool
        """
        # Create generic sources if None are passed
        if srcs is None:
            srcs = [str(i) for i in range(data.shape[0])]
        # Make uniform source reliabilities if none where passed
        if src_reliabilities is None:
            src_reliabilities = [1 for _ in range(len(srcs))]
        # Construct best inferences (bayesian knowledge fragments)
        self.bkfs, self.report = self.backend.learn(data, feature_states, verbose=verbose)
        # Fuse fragments
        self.report.start_timer()
        self.learned_bkb = fuse(self.bkfs, src_reliabilities, source_names=srcs, collapse=collapse)
        self.report.fusion_time = self.report.end_timer()
        self.report.add_learned_bkb_scores(
                *self.learned_bkb.score(
                    data,
                    feature_states,
                    self.backend.score,
                    store=self.backend.store,
                    only='both',
                    is_learned=True,
                    )
                )   
        # Finalize Report
        self.report.finalize()
        return

class BNLearner:
    def __init__(
            self,
            backend:str,
            score:str,
            palim:int=None,
            distributed:bool=False,
            ray_address:str='auto',
            ) -> None:
        """ The Bayesian Network Learner class.

        Args:
            :param backend: Name of the learning backend. Choices: gobnilp or notears.
            :type name: str
            :param score: Name of scoring function. Choices: mdl_mi, mdl_ent.
            :type score: str

        Kwargs:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param distributed: Whether to distrbute learning across a Ray cluster.
            :type distributed: bool
            :param ray_address: The address of the Ray cluster. Defaults to auto.
            :type ray_address: str
        """
        if backend == 'gobnilp':
            if distributed:
                self.backend = BNGobnilpDistributedBackend(score, palim, ray_address)
            else:
                self.backend = BNGobnilpBackend(score, palim)
        elif backend == 'notears':
            if distributed:
                self.backend = NoTearsDistributedBackend(score)
            else:
                self.backend = NoTearsBackend(score)
        else:
            raise ValueError(f'Unknown backend: {backend}.')

    def fit(
            self,
            data:np.array,
            feature_states:list,
            verbose:bool=False,
            ) -> None:
        """ Learn a BN from data.

        Args:
            :param data: NxM binary matrix where N is the number of data instances and M
            is the number of feature instantiations. Data instance row should have a 1 if 
            the column feature instantiation is present in the instance and zero otherwise.
            :type data: np.array
            :param feature_states: A list of feature instantiation tuples of the form: [(feature1, state1), (feature1, state2),...].
            Must match the M dimension of the data exactly.
            :type feature_states: list
        """
        self.bn, self.m, self.report = self.backend.learn(data, feature_states, verbose=verbose)
        self.report.score = self.m.learned_scores[0]
        return
