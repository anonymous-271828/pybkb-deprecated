import ray
import os
import numpy as np
import itertools
import logging
import contextlib
import time
import tqdm
from operator import itemgetter
from pygobnilp.gobnilp import Gobnilp
from gurobipy import GRB
from ray.util.placement_group import placement_group
from ray.util import ActorPool
from ray import workflow

from pybkb.utils.probability import *
from pybkb.utils.mp import MPLogger
from pybkb.scores import *
from pybkb.learn.report import LearningReport
from pybkb.bkb import BKB
from pybkb.bn import BN


class BKBGobnilpBackend:
    def __init__(
            self,
            score:str,
            palim:int=None,
            only:str=None,
            ) -> None:
        """ BKB Gobnilp DAG learning backend.

        Args:
            :param score: Name of scoring function. Choices: mdl_mi, mdl_ent.
            :type score: str

        Kwargs:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
        """
        if score == 'mdl_mi':
            self.score_node = MdlMutInfoScoreNode
        elif score == 'mdl_ent':
            self.score_node = MdlEntScoreNode
        else:
            raise ValueError(f'Unknown score: {score}')
        self.palim = palim
        self.score = score
        self.only = only 
        self.store = build_probability_store()

    def calculate_all_local_scores(
            self,
            data:np.array,
            feature_states:list,
            filepath:str=None,
            ) -> dict:
        """ Generates local scores for Gobnilp optimization
        
        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list

        Kwargs:
            :param filepath: Optional filepath where local scores will be written. Defaults None.
            :type filepath: str
        """
        # Build feature states index map
        feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        # Reset store
        self.store = build_probability_store()
        # Calculate scores
        for data_idx in range(data.shape[0]):
            scores[data_idx] = self.calculate_local_score(
                    data_idx,
                    data,
                    feature_states,
                    feature_states_index_map,
                    )
        return scores

    @staticmethod
    def calculate_local_score_static(
            data_idx,
            data:np.array,
            feature_states:list,
            palim,
            score_node,
            feature_states_index_map:dict,
            filepath:str=None,
            store:dict=None,
            only:str=None,
            ):
        """ Generates a data instance's local score for Gobnilp optimization. Static method to be used externally.
        
        Args:
            :param data_idx: Row index of data instance to calculate local scores.
            :type data_idx: int
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param feature_states_index_map: A dictionary mapping feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map: dict
            :param palim: Parent set limit. None means that there is no limit.
            :type palim: int
            :param score_node: The score node object.
            :type score_node: pybkb.learn.scores.ScoreNode

        Kwargs:
            :param filepath: Optional filepath where local scores will be written. Defaults None.
            :type filepath: str
            :param store: The probability store for the various joint probability calculations.
            :type store: dict
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
        """
        # Collect feature instantiations in this data instance
        fs_indices = np.argwhere(data[data_idx,:] == 1).flatten()
        # Initialize scores
        scores = defaultdict(dict)
        # Initialize store if not passed
        if store is None:
            store = build_probability_store()
        # Calculate node encoding length
        node_encoding_len = np.log2(len(feature_states))
        # Calculate Scores
        for fs_idx in fs_indices:
            # Need to add one to palim due to python zero indexing
            for i in range(palim + 1):
                # No parent set score
                if i == 0:
                    node = score_node(fs_idx, node_encoding_len)
                    score, store = node.calculate_score(
                            data,
                            feature_states,
                            store,
                            feature_states_index_map=feature_states_index_map,
                            only=only,
                            )
                    # Need to cast index as str for gobnilp
                    scores[str(fs_idx)][frozenset()] = score
                    continue
                # For non empty parent sets
                for pa_set in itertools.combinations(set.difference(set(fs_indices), {fs_idx}), r=i):
                    node = score_node(fs_idx, node_encoding_len, pa_set=pa_set)
                    score, store = node.calculate_score(
                            data,
                            feature_states,
                            store,
                            feature_states_index_map=feature_states_index_map,
                            only=only,
                            )
                    # Need to cast index as str for gobnilp
                    scores[str(fs_idx)][frozenset([str(pa) for pa in pa_set])] = score
        if filepath:
            # Make into string format
            s = f'{len(features)}\n'
            for feature, pa_scores in scores.items():
                s += f'{feature} {len(pa_scores)}\n'
                for pa_set, score in pa_scores.items():
                    if pa_set is None:
                        pa_set = []
                    s += f'{score} {len(pa_set)}'
                    for pa in pa_set:
                        s += f' {pa}'
                    s += '\n'
            # Write to file
            with open(filepath, 'w') as f_:
                f_.write(s)
        return dict(scores), store

    def calculate_local_score(
            self,
            data_idx:int,
            data:np.array,
            feature_states:list,
            feature_states_index_map:dict,
            filepath:str=None,
            ) -> dict:
        """ Generates a data instance's local score for Gobnilp optimization
        
        Args:
            :param data_idx: Row index of data instance to calculate local scores.
            :type data_idx: int
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param feature_states_index_map: A dictionary mapping feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map: dict

        Kwargs:
            :param filepath: Optional filepath where local scores will be written. Defaults None.
            :type filepath: str
        """
        score, self.store = self.calculate_local_score_static(
                data_idx,
                data,
                feature_states,
                self.palim,
                self.score_node,
                feature_states_index_map,
                filepath,
                self.store,
                self.only,
                )
        return score

    @staticmethod
    def convert_dag_to_bkf(dag, name, data, feature_states, feature_states_index_map, store):
        """ Converts DAG learned by Gobnilp to a BKF inference fragment.
        """
        bkf = BKB(name)
        for node_idx_str in dag.nodes:
            # Gobnilp names are strings so need recast as int
            node_idx = int(node_idx_str)
            # Get head feature name and state name
            head_feature, head_state = feature_states[node_idx]
            # Add to BKF
            bkf.add_inode(head_feature, head_state)
            # Collect all incident nodes to build the tail
            tail_indices = []
            tail = []
            for edge in dag.in_edges(node_idx_str):
                tail_node_idx = int(edge[0])
                # Get tail feature and state name
                tail_feature, tail_state = feature_states[tail_node_idx]
                # Add to BKF
                bkf.add_inode(tail_feature, tail_state)
                # Collect tail
                tail.append((tail_feature, tail_state))
                tail_indices.append(tail_node_idx)
            # Calculate S-node conditional probability
            prob, store = joint_prob(data, node_idx, tail_indices, store)
            # Make S-node
            bkf.add_snode(head_feature, head_state, prob, tail)
        return bkf

    def learn(self, data:np.array, feature_states:list, verbose:bool=False):
        """ Learns the best set of BKFs from the data.

        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
        """
        # Initialize report
        report = LearningReport('gobnilp', False)
        report.initialize_bkf_metrics(data.shape[0])
        # Build feature states index map
        feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        # Reset store
        self.store = build_probability_store()
        # Update palim if necessary
        if self.palim is None:
            self.palim = len(set([f for f, s in feature_states])) - 1
        # Build optimal bkfs
        bkfs = []
        for data_idx in tqdm.tqdm(range(data.shape[0]), desc='Learning Fragments', disable=not verbose):
            # Calculate local scores
            scores = self.calculate_local_score(data_idx, data, feature_states, feature_states_index_map)
            # Update report
            report.update_from_bkf_store(data_idx, self.store)
            # Learn the best DAG from these local scores using Gobnilp
            # Redirect output so we don't have to see this
            f = open(os.devnull, 'w')
            with contextlib.redirect_stdout(f):
                m = Gobnilp()
                # Start the learning but stop before learning to add constraints
                m.learn(local_scores_source=scores, end='MIP model')
                # Grab all the adjacency variables
                adj = [v for p, v in m.adjacency.items()]
                # Add a constraint that at the DAG must be connected (need to subtract one to make a least a tree)
                m.addLConstr(sum(adj), GRB.GREATER_EQUAL, np.sum(data[data_idx,:]) - 1)
            # Close devnull file as to not get resource warning
            f.close()
            # Learn the DAG
            report.start_timer()
            m.learn(local_scores_source=scores, start='MIP model')
            # Add learning time to report
            report.add_bkf_metrics(data_idx, learn_time=report.end_timer())
            # Convert learned DAG to BKF (learned_bn of gobnilp is a subclass of networkx.DiGraph)
            bkfs.append(
                    self.convert_dag_to_bkf(
                        m.learned_bn,
                        str(data_idx),
                        data,
                        feature_states,
                        feature_states_index_map,
                        self.store
                        )
                    )
            # Get scores for this bkf to put in report
            _dscore, _mscore = bkfs[-1].score(
                    data,
                    feature_states,
                    self.score,
                    feature_states_index_map=feature_states_index_map, 
                    only='both',
                    store=self.store,
                    )
            report.add_bkf_metrics(
                    data_idx,
                    model_score=_mscore,
                    data_score=_dscore,
                    bn=m.learned_bn,
                    )
        return bkfs, report


class BKBGobnilpDistributedBackend(BKBGobnilpBackend):
    def __init__(
            self,
            score:str,
            #num_learners:int,
            #num_cluster_nodes:int,
            palim:int=None,
            only:str=None,
            ray_address:str=None,
            ) -> None:
        """ BKB Gobnilp DAG learning backend that distributes learning over a ray cluster

        Args:
            :param score: Name of scoring function. Choices: mdl_mi, mdl_ent.
            :type score: str
            :param num_learners: Number of ray learner workers to use on each node.
            :type num_learners: int 
            :param num_cluster_nodes: Number of ray nodes that are in the cluster.
            :type num_learners: int 

        Kwargs:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
            :param ray_address: The address of the Ray cluster. Defaults to auto.
            :type ray_address: str
        """
        if ray_address is None:
            print('Warning: You did not pass a ray address so assuming ray has already been initialized.')
        self.ray_address = ray_address
        super().__init__(score, palim, only)

    def setup_cluster(self):
        """ Function to initialize the ray cluster and bundle workers onto each cluster node to
        optimize for space.
        """
        # Initialize
        if self.ray_address is not None:
            ray.init(address=self.ray_address)
        ## Create bundles
        #bundles = [{"CPU": self.num_learners} for _ in range(self.num_cluster_nodes)]
        ## Create Placement Group
        #pg = placement_group(bundles, strategy='STRICT_SPREAD')
        ## Wait for the placement group to be ready
        #ray.get(pg.ready())
        #return bundles, pg
        return 

    def setup_actor_pool(self, pg, bundles):
        """ Sets up the actor learner pool.
        """
        return ActorPool(
                [
                    BKBGobnilpLearner.options(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_idx,
                        ).remote(
                            self.score,
                            self.data_id,
                            self.feature_states_id,
                            self.feature_states_index_map_id,
                            self.palim,
                            self.only,
                            ) for bundle_idx in range(len(bundles))
                    ]
                )

    def put_data_on_cluster(self, data, feature_states, feature_states_index_map):
        """ Will put all sharable data into the ray object store.
        """
        self.data_id = ray.put(data)
        self.feature_states_id = ray.put(np.array(feature_states))
        self.feature_states_index_map_id = ray.put(feature_states_index_map)
        return 

    def setup_store(self, data, verbose):
        """ Will calculate all the unique probability values needed to compute later scores
        in order to reduce repeat calculations.
        """
        unique_joints_to_calc = set([frozenset()])
        for data_idx in tqdm.tqdm(range(data.shape[0]), desc='Extracting probabilities', disable=not verbose, leave=False):
            # Collect feature instantiations in this data instance
            fs_indices = np.argwhere(data[data_idx,:] == 1).flatten()
            for fs_idx in fs_indices:
                for i in range(self.palim + 1):
                    # No parent set score
                    if i == 0:
                        unique_joints_to_calc.add(frozenset([fs_idx]))
                        continue
                    # For non empty parent sets
                    for pa_set in itertools.combinations(set.difference(set(fs_indices), {fs_idx}), r=i):
                        unique_joints_to_calc.add(frozenset([fs_idx] + list(pa_set)))
        # Calculate all unique joint probabilities and create an aggregated store
        store_ids = []
        for joint_indices in tqdm.tqdm(unique_joints_to_calc, desc='Setting up remote calls', disable=not verbose, leave=False):
            store_ids.append(calc_joint_prob.remote(self.data_id, joint_indices))
        return store_ids

    def setup_work(self, data_len, logger, store):
        bkf_ids, hashlookup_ids, ncalls_ids = [], [], []
        for data_idx in range(data_len):
            # Calculate Scores
            score_id, ncalls_id, hashlookups_id = calculate_local_score.remote(
                data_idx,
                self.data_id,
                self.feature_states_id,
                self.palim,
                self.score_node,
                self.feature_states_index_map_id,
                store,
                self.only,
                )
            
            # Build BKF
            bkf_id = learn_bkf_structure.remote(
                        data_idx,
                        score_id,
                        self.data_id,
                        self.feature_states_id,
                        self.feature_states_index_map_id,
                        store,
                        )
            bkf_ids.append(bkf_id)
            ncalls_ids.append(ncalls_id)
            hashlookup_ids.append(hashlookups_id)
        return bkf_ids, ncalls_ids, hashlookup_ids
        
    def learn(self, data, feature_states, verbose:bool=False):
        """ Learns the best set of BKFs from the data.

        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
        """
        # Update palim if necessary
        if self.palim is None:
            self.palim = len(set([f for f, s in feature_states])) - 1
        report = LearningReport('gobnilp', False)
        report.initialize_bkf_metrics(data.shape[0])
        logger = MPLogger('GobnilpDistributedBackend', logging.INFO, loop_report_time=60)
        # Build feature states index map
        feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        # Setup cluster
        #bundles, pg = self.setup_cluster()
        self.setup_cluster()
        logger.info('Setup cluster.')
        # Put data into ray object store
        self.put_data_on_cluster(
                data, 
                feature_states,
                feature_states_index_map,
                )
        logger.info('Put data into object store.')
        # Setup store
        logger.info('Setting up probability store.')
        store_ids = self.setup_store(data, verbose)
        logger.info('Calculating probabilities necessary for scores.')
        # Calculate necessary joint probabilities
        total = len(store_ids)
        i = 0
        logger.initialize_loop_reporting()
        stores = []
        while len(store_ids):
            i += 1
            done_ids, store_ids = ray.wait(store_ids)
            stores.append(ray.get(done_ids[0]))
            logger.report(i=i, total=total)
        store = self.aggregate_stores(stores, verbose)
        store_id = ray.put(store)
        logger.info('Finished probability calculations.')
        logger.info('Setting up work.')
        bkf_ids, ncalls_ids, hashlookup_ids = self.setup_work(data.shape[0], logger, store_id)
        logger.info('Completed setup.')
        # Run work
        bkfs = []
        bkfs_finished = 0
        total = len(bkf_ids)
        logger.info('Collecting BKFs...')
        logger.initialize_loop_reporting()
        while len(bkf_ids):
            done_ids, bkf_ids = ray.wait(bkf_ids)
            bkf_obj, learn_time = ray.get(done_ids[0])
            bkf, report = self.postprocess_bkf(
                    bkf_obj,
                    learn_time,
                    report,
                    data,
                    feature_states,
                    feature_states_index_map,
                    store,
                    )
            bkfs_finished += 1
            logger.report(i=bkfs_finished, total=total)
            bkfs.append(bkf)
        # Collect lookup savings
        store_calls = ray.get(ncalls_ids + hashlookup_ids)
        store['__ncalls__'] = sum(store_calls[:int(len(store_calls)/2)]) + len(stores)
        store['__nhashlookups__'] = sum(store_calls[int(len(store_calls)/2):])
        report.update_from_store(store)
        # Sort bkfs into correct order based on data index
        bkfs_to_sort = [(int(bkf.name), bkf) for bkf in bkfs]
        bkfs = [bkf for i, bkf in sorted(bkfs_to_sort, key=itemgetter(0))]
        return bkfs, report

    def aggregate_stores(self, stores:list, verbose):
        store = build_probability_store()
        for _store in tqdm.tqdm(stores, desc='Aggregating stores', disable=not verbose, leave=False):
            # Update probabilities only
            store.update({key: value for key, value in _store.items() if '__' not in key})
        return store

    def postprocess_bkf(
            self,
            bkf_obj,
            learn_time,
            report,
            data,
            feature_states,
            feature_states_index_map,
            store,
            ):
        # Load bkf object
        bkf = BKB.loads(bkf_obj)
        data_idx = int(bkf.name)
        # Calculate metrics
        report.add_bkf_metrics(data_idx, learn_time=learn_time)
        # Get scores for this bkf to put in report
        _dscore, _mscore = bkf.score(
                data,
                feature_states,
                self.score,
                feature_states_index_map=feature_states_index_map, 
                only='both',
                store=store,
                )
        report.add_bkf_metrics(
                data_idx,
                model_score=_mscore,
                data_score=_dscore,
                )
        return bkf, report

    def learn_old(self, data, feature_states, verbose:bool=False):
        """ Learns the best set of BKFs from the data.

        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
        """
        report = LearningReport('gobnilp', False)
        report.initialize_bkf_metrics(data.shape[0])
        logger = MPLogger('GobnilpDistributedBackend', logging.INFO, loop_report_time=60)
        # Build feature states index map
        feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        # Setup cluster
        bundles, pg = self.setup_cluster()
        logger.info('Setup cluster.')
        # Put data into ray object store
        self.put_data_on_cluster(
                data, 
                feature_states,
                feature_states_index_map,
                )
        logger.info('Put data into object store.')
        # Initialize actor pool
        pool = self.setup_actor_pool(pg, bundles)
        # Setup work
        res_ids = pool.map_unordered(lambda a, data_idx: a._learn_bkf.remote(data_idx), range(data.shape[0]))
        logger.info('Placed all work.')
        # Collect
        bkfs = []
        workers_finished = 0
        total_workers = self.num_learners * len(bundles)
        logger.info('Collecting...')
        logger.initialize_loop_reporting()
        for res in res_ids:
            workers_finished += 1
            bkf_obj = res
            bkfs.append(BKB.loads(bkf_obj))
            logger.report(workers_finished, total=data.shape[0])
        # Sort bkfs into correct order based on data index
        bkfs_to_sort = [(int(bkf.name), bkf) for bkf in bkfs]
        bkfs = [bkf for i, bkf in sorted(bkfs_to_sort, key=itemgetter(0))]
        return bkfs, report


@ray.remote(num_cpus=1)
class BKBGobnilpLearner(BKBGobnilpBackend):
    def __init__(
            self,
            score:str,
            data_id,
            feature_states_id,
            feature_states_index_map_id,
            palim:int=None,
            only:str=None,
            ) -> None:
        """ BKB Gobnilp DAG learner that operates on the ray cluster.

        Args:
            :param score: Name of scoring function. Choices: mdl_mi, mdl_ent.
            :type score: str

        Kwargs:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
        """
        super().__init__(score, palim, only)
        self.data_id = data_id
        self.feature_states_id = feature_states_id
        self.feature_states_index_map_id = feature_states_index_map_id

    def _learn_bkf(self, data_idx):
        # Calculate local scores
        scores = self.calculate_local_score(data_idx, self.data_id, self.feature_states_id, self.feature_states_index_map_id)
        # Learn the best DAG from these local scores using Gobnilp
        # Redirect output so we don't have to see this
        f = open(os.devnull, 'w')
        with contextlib.redirect_stdout(f):
            m = Gobnilp()
            # Start the learning but stop before learning to add constraints
            m.learn(local_scores_source=scores, end='MIP model')
            # Grab all the adjacency variables
            adj = [v for p, v in m.adjacency.items()]
            # Add a constraint that at the DAG must be connected
            m.addConstr(sum(adj), GRB.GREATER_EQUAL, np.sum(self.data_id[data_idx,:]))
        # Close devnull file as to not get resource warning
        f.close()
        # Learn the DAG
        m.learn(local_scores_source=scores, start='MIP model')
        # Convert learned DAG to BKF (learned_bn of gobnilp is a subclass of networkx.DiGraph)
        bkf = self.convert_dag_to_bkf(
                m.learned_bn,
                str(data_idx),
                self.data_id,
                self.feature_states_id,
                self.feature_states_index_map_id,
                )
        # Transmit BKF efficiently using inherent numpy representation and reload on other end
        return bkf.dumps()


class BNGobnilpBackend:
    def __init__(
            self,
            score:str,
            palim:int=None,
            only:str=None,
            ) -> None:
        """ BKB Gobnilp DAG learning backend.

        Args:
            :param score: Name of scoring function. Choices: mdl_mi, mdl_ent.
            :type score: str

        Kwargs:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
        """
        if score == 'mdl_mi':
            self.score_node = MdlMutInfoScoreNode
        elif score == 'mdl_ent':
            self.score_node = MdlEntScoreNode
        else:
            raise ValueError(f'Unknown score: {score}')
        self.palim = palim
        self.score = score
        self.only = only 
        self.store = {}

    def calculate_all_local_scores(
            self,
            data:np.array,
            features:set,
            states:dict,
            feature_states_map:dict,
            feature_states:list,
            filepath:str=None,
            verbose:bool=False,
            ) -> dict:
        """ Generates local scores for Gobnilp optimization
        
        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param features: A set of feature names.
            :type features: set
            :param states: A dictionary of states for each feature. Differs from feature_states_map
                because it doesn't contain the index of in the feature_states.
            :type states: dict
            :param feature_states_map: A dictionary keyed by feature name with 
                values equalling a list of allowed states of the form [(idx, state_name), ...].
                Use the pybkb.utils.probability.build_feature_state_map function to build this map.
            :type feature_states_map: dict
            :param feature_states: List of feature instantiations.
            :type feature_states: list

        Kwargs:
            :param filepath: Optional filepath where local scores will be written. Defaults None.
            :type filepath: str
        """
        node_encoding_len = np.log2(len(features))
        # Setup parent set limit if None
        if self.palim is None:
            palim = len(features) - 1
        else:
            palim = self.palim
        # Initialize scores
        scores = defaultdict(dict)
        # Calculate MDL scores
        for feature in tqdm.tqdm(features, desc='Scoring', disable=not verbose, leave=False):
            for i in range(palim):
                if i == 0:
                    node = self.score_node(
                            feature,
                            node_encoding_len,
                            states=states,
                            indices=False,
                            rv_level=True,
                            )
                    score, self.store = node.calculate_score(
                            data,
                            feature_states,
                            self.store,
                            feature_states_map=feature_states_map,
                            only=self.only,
                            )
                    scores[feature][frozenset()] = score
                    continue
                for pa_set in itertools.combinations(set.difference(features, {feature}), r=i):
                    node = self.score_node(
                            feature,
                            node_encoding_len,
                            pa_set=pa_set,
                            states=states,
                            indices=False,
                            rv_level=True,
                            )
                    score, self.store = node.calculate_score(
                            data,
                            feature_states,
                            self.store,
                            feature_states_map=feature_states_map,
                            only=self.only,
                            )
                    scores[feature][frozenset(pa_set)] = score
        if filepath:
            # Make into string format
            s = f'{len(features)}\n'
            for feature, pa_scores in scores.items():
                s += f'{feature} {len(pa_scores)}\n'
                for pa_set, score in pa_scores.items():
                    if pa_set is None:
                        pa_set = []
                    s += f'{score} {len(pa_set)}'
                    for pa in pa_set:
                        s += f' {pa}'
                    s += '\n'
            # Write to file
            with open(filepath, 'w') as f_:
                f_.write(s)
        return dict(scores)

    def learn(self, data:np.array, feature_states:list, verbose:bool=False):
        """ Learns the best BN from the data.

        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
        """
        # Initialize report
        report = LearningReport('gobnilp', True)
        # Collect features and states
        features = []
        states = defaultdict(list)
        feature_states_map = build_feature_state_map(feature_states)
        for f, s in feature_states:
            features.append(f)
            states[f].append(s)
        features = set(features)
        states = dict(states)
        # Reset store
        self.store = build_probability_store()
        # Calculate local scores
        scores = self.calculate_all_local_scores(data, features, states, feature_states_map, feature_states, verbose=verbose)
        # Update report with calls to joint calculator
        report.update_from_store(self.store)
        # Learn the best DAG from these local scores using Gobnilp
        m = Gobnilp()
        # Start the learning but stop before learning to add constraints
        m.learn(local_scores_source=scores, end='MIP model')
        # Grab all the adjacency variables
        adj = [v for p, v in m.adjacency.items()]
        # Add a constraint that at the DAG must be connected
        m.addLConstr(sum(adj), GRB.GREATER_EQUAL, len(features) - 1)
        # Learn the DAG
        report.start_timer()
        m.learn(local_scores_source=scores, start='MIP model')
        report.add_bn_learn_time(report.end_timer())
        # Convert learned_bn from pygobnilp to interal bn representation so we can score like a BKB
        bn = BN.from_bnlearn_modelstr(m.learned_bn.bnlearn_modelstring(), states)
        data_score, model_score = bn.score_like_bkb(
                data,
                feature_states,
                self.score,
                feature_states_map,
                only='both',
                store=self.store,
                )
        #print(bn.json())
        report.add_bn_like_bkb_scores(data_score, model_score)
        # Finialize report
        report.finalize()
        return m.learned_bn, m, report

#### Distrbuted Remote Functions

@ray.remote
def calc_joint_prob(data_id, joint_indices):
    _, store = joint_prob(data_id, parent_state_indices=list(joint_indices))
    return store

@ray.remote(num_returns=3)
def calculate_local_score(
        data_idx,
        data_id,
        feature_states_id,
        palim,
        score_node,
        feature_states_index_map_id,
        store,
        only:str=None,
        ):
    """ Gobnilp local score wrapper for ray workflow.
    """
    # Calculate scores and store
    scores, store = BKBGobnilpBackend.calculate_local_score_static(
            data_idx,
            data_id,
            feature_states_id,
            palim,
            score_node,
            feature_states_index_map_id,
            store=store,
            only=only,
            )
    # Put them in the object store
    return scores, store['__ncalls__'], store['__nhashlookups__']

@ray.remote(num_cpus=1)
def learn_bkf_structure(
        data_idx,
        score_id,
        data_id,
        feature_states_id,
        feature_states_index_map_id,
        store_id,
        ):
    # Learn the best DAG from these local scores using Gobnilp
    # Redirect output so we don't have to see this
    f = open(os.devnull, 'w')
    with contextlib.redirect_stdout(f):
        m = Gobnilp()
        # Start the learning but stop before learning to add constraints
        m.learn(local_scores_source=score_id, end='MIP model')
        # Grab all the adjacency variables
        adj = [v for p, v in m.adjacency.items()]
        # Add a constraint that at the DAG must be connected
        m.addConstr(sum(adj), GRB.GREATER_EQUAL, np.sum(data_id[data_idx,:]) - 1)
    # Close devnull file as to not get resource warning
    f.close()
    # Learn the DAG
    start_time = time.time()
    m.learn(local_scores_source=score_id, start='MIP model')
    learn_time = time.time() - start_time
    # Convert learned DAG to BKF (learned_bn of gobnilp is a subclass of networkx.DiGraph)
    bkf = BKBGobnilpBackend.convert_dag_to_bkf(
            m.learned_bn,
            str(data_idx),
            data_id,
            feature_states_id,
            feature_states_index_map_id,
            store=store_id,
            )
    # Transmit BKF efficiently using inherent numpy representation and reload on other end
    return (bkf.dumps(), learn_time)
