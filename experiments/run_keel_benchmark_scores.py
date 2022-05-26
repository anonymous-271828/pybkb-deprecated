import os
import ray
import logging
import tqdm
import compress_pickle
import argparse
from operator import itemgetter
from ray.util.placement_group import placement_group
from collections import defaultdict

from pybkb.learn import BKBLearner, BNLearner
from pybkb.utils.data import KeelWrangler
from pybkb.utils.mp import MPLogger
from pybkb.utils.probability import build_probability_store, build_feature_state_map


## Functions
def setup_cluster(num_nodes, num_workers_per_node):
    # Initialize Ray
    ray.init(address='auto')
    # Setup placement group and bundles
    bundles = [{"CPU": num_workers_per_node} for _ in range(num_nodes)]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    return pg, bundles

## Remote Functions
@ray.remote(num_cpus=1)
def get_scores(dataset_path, scores_path, max_palim):
    # Get dataset name
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    # Setup logger
    logger = MPLogger(dataset_name, logging.INFO, loop_report_time=60)
    # Initialize wrangler and load data
    logger.info('Loading data...')
    with open(dataset_path, 'rb') as f_:
        data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
    # Collect features and states
    features = []
    states = defaultdict(list)
    feature_states_map = build_feature_state_map(feature_states)
    for f, s in feature_states:
        features.append(f)
        states[f].append(s)
    features = set(features)
    states = dict(states)
    # Change palim if it exceeds number of features in the dataset
    max_palim = min(max_palim, len(features) - 1)
    logger.info('Data loaded.')
    # Make full paths
    result_path_bkb = os.path.join(scores_path, 'bkb', dataset_name)
    result_path_bn = os.path.join(scores_path, 'bn', dataset_name)
    if not os.path.exists(result_path_bkb):
        os.makedirs(result_path_bkb)
    if not os.path.exists(result_path_bn):
        os.makedirs(result_path_bn)
    store_bkb = build_probability_store()
    store_bn = build_probability_store()
    all_scores_bkb = {}
    all_scores_bn = {}
    for palim in range(1, max_palim+1):
        # Setup learner
        bkb_learner = BKBLearner(
                'gobnilp',
                'mdl_ent',
                distributed=False, 
                palim=palim,
                )
        bn_learner = BNLearner(
                'gobnilp',
                'mdl_ent',
                palim=palim,
                )
        # Add in all the calculations we've previously done
        bkb_learner.backend.store = store_bkb
        bkb_learner.backend.all_scores = all_scores_bkb
        bn_learner.backend.store = store_bn
        bn_learner.backend.all_scores = all_scores_bn
        # Collect scores
        logger.info(f'Collecting scores for palim = {palim}...')
        logger.info('Collecting scores for BKB.')
        logger.initialize_loop_reporting()
        scores_bkb = bkb_learner.backend.calculate_all_local_scores(
                data,
                feature_states,
                verbose=False,
                logger=logger,
                reset=False,
                )
        logger.info('Collecting scores for BN.')
        logger.initialize_loop_reporting()
        scores_bn = bn_learner.backend.calculate_all_local_scores(
                data,
                features,
                states,
                feature_states_map,
                feature_states,
                verbose=False,
                logger=logger,
                reset=False,
                )
        # Add just calculated all scores and store
        store_bkb = bkb_learner.backend.store
        all_scores_bkb = bkb_learner.backend.all_scores
        store_bn = bn_learner.backend.store
        all_scores_bn = bn_learner.backend.all_scores
        # Save out store and scores
        logger.info('Saving out scores...')
        with open(os.path.join(result_path_bkb, f'palim-{palim}.scores'), 'wb') as scores_file:
            compress_pickle.dump(scores_bkb, scores_file, compression='lz4')
        with open(os.path.join(result_path_bn, f'palim-{palim}.scores'), 'wb') as scores_file:
            compress_pickle.dump(scores_bn, scores_file, compression='lz4')
        logger.info('Scores saved.')
        logger.info('Saving out store.')
        with open(os.path.join(result_path_bkb, f'palim-{palim}.store'), 'wb') as store_file:
            compress_pickle.dump(bkb_learner.backend.store, store_file, compression='lz4')
        with open(os.path.join(result_path_bn, f'palim-{palim}.store'), 'wb') as store_file:
            compress_pickle.dump(bn_learner.backend.store, store_file, compression='lz4')
        # Delete scores
        del scores_bkb
        del scores_bn
    # Delete stores and all_scores
    del store_bkb
    del all_scores_bkb
    del store_bn
    del all_scores_bn
    logger.info('Complete.')

def main(num_nodes, num_workers_per_node, datasets_path, results_path, dataset_name=None, max_palim=10):
    if dataset_name is None:
        print('Running all datasets. Will take awhile.')
        datasets = [d for d in os.listdir(datasets_path) if 'no_missing_values' in d]
    else:
        datasets = [dataset_name]
    # Setup logging
    logger = MPLogger('Main', logging.INFO)
    # Setup Cluster
    pg, bundles = setup_cluster(num_nodes, num_workers_per_node)
    # Setup score collection work
    score_ids = []
    bundle_idx = 0
    dataset_counter = 0
    for dataset in datasets:
        dataset_counter += 1
        if dataset_counter > num_workers_per_node:
            # Move on to the next bundle
            dataset_counter = 0
            bundle_idx += 1
        # Make full paths
        dataset_path = os.path.abspath(os.path.join(datasets_path, dataset))
        results_path = os.path.abspath(results_path)
        score_ids.append(
                get_scores.options(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx
                    ).remote(
                        dataset_path,
                        results_path,
                        max_palim,
                        )
                    )

    i = 0
    total = len(score_ids)
    logger.initialize_loop_reporting()
    while len(score_ids):
        done_ids, score_ids = ray.wait(score_ids)
        _ = ray.get(done_ids[0])
        i += 1
        logger.report(i=i, total=total)
    logger.info('Complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Master path to all preprocessed BKB datasets.')
    parser.add_argument('results_path', help='Path to directory where results should be saved.')
    parser.add_argument('--dataset_name', help='Name of dataset to calculate scores.', default=None)
    parser.add_argument('--num_nodes', help='Number of nodes to distribute over. Should match Ray cluster.', default=1)
    parser.add_argument('--num_workers_per_node', help='Number of workers that should be placed on each node.',default=5)
    parser.add_argument('--max_palim', help='Maximum Parset Limit that should be iterated to.',default=10)
    args = parser.parse_args()

    # Run main script
    main(
            args.num_nodes,
            args.num_workers_per_node,
            args.dataset_path,
            args.results_path,
            args.dataset_name,
            args.max_palim,
            )
