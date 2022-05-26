import os
import ray
import logging
import tqdm
import json
import compress_pickle
from operator import itemgetter
from ray.util.placement_group import placement_group
from collections import defaultdict

from pybkb.learn import BKBLearner, BNLearner
from pybkb.utils.data import KeelWrangler
from pybkb.utils.mp import MPLogger
from pybkb.utils.probability import build_probability_store, build_feature_state_map

## Script Parameters
NUM_NODES = 15
WORKERS_PER_NODE = 5
# Set datasets path
cwd = os.path.dirname(__file__)
datasets_path = os.path.join(cwd, '../', 'data/keel/bkb_data/cross_validation_sets')

# Collect highest palim that are common between both bn and bkb score runs
# We already saved these results in a file, edit this file or make your own based on what you'd like to test
with open('palim_common_options.json', 'r') as f_:
    common_options = json.load(f_)

## Functions
def setup_cluster():
    # Initialize Ray
    ray.init(address='auto')
    # Setup placement group and bundles
    bundles = [{"CPU": WORKERS_PER_NODE} for _ in range(NUM_NODES)]
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
    logger.info('Starting cross validation scoring...')
    for idx, cross_valid_set in enumerate(os.listdir(dataset_path)):
        cross_valid_path = os.path.join(dataset_path, cross_valid_set)
        cross_valid_name = cross_valid_set.split('.')[0]
        logger.info(f'Starting cross validation set {idx}.')
        # Load cross validation data set
        with open(cross_valid_path, 'rb') as f_:
            # format is (data_train, data_test, feature_states, srcs_train, srcs_test)
            data, _, feature_states, srcs, _ = compress_pickle.load(f_, compression='lz4')
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
        logger.info('Data loaded.')
        # Make full paths
        result_path_bkb = os.path.join(scores_path, 'bkb', dataset_name)
        result_path_bn = os.path.join(scores_path, 'bn', dataset_name)
        if not os.path.exists(result_path_bkb):
            os.makedirs(result_path_bkb)
        if not os.path.exists(result_path_bn):
            os.makedirs(result_path_bn)
        # Setup learners
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
        logger.info('Collecting scores for BKB.')
        logger.initialize_loop_reporting()
        scores_bkb = bkb_learner.backend.calculate_all_local_scores(
                data,
                feature_states,
                verbose=False,
                logger=logger,
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
                )
        # Save out store and scores
        logger.info('Saving out scores...')
        with open(os.path.join(result_path_bkb, f'{cross_valid_name}.scores'), 'wb') as scores_file:
            compress_pickle.dump(scores_bkb, scores_file, compression='lz4')
        with open(os.path.join(result_path_bn, f'{cross_valid_name}.scores'), 'wb') as scores_file:
            compress_pickle.dump(scores_bn, scores_file, compression='lz4')
        logger.info('Scores saved.')
        # Delete scores
        del scores_bkb
        del scores_bn
        logger.info(f'Finished cross validation set {idx}.')
    logger.info('Complete.')

# Setup logging
logger = MPLogger('Main', logging.INFO)

# Setup Cluster
pg, bundles = setup_cluster()

# Setup score collection work
score_ids = []
bundle_idx = 0
dataset_counter = 0
for dataset, palim in common_options.items():
    dataset_counter += 1
    if dataset_counter > WORKERS_PER_NODE:
        # Move on to the next bundle
        dataset_counter = 0
        bundle_idx += 1
    # Make full paths
    dataset_path = os.path.abspath(os.path.join(datasets_path, dataset))
    results_path = os.path.abspath(os.path.join(cwd, 'scores/cross_validation'))
    score_ids.append(
            get_scores.options(
                placement_group=pg,
                placement_group_bundle_index=bundle_idx
                ).remote(
                    dataset_path,
                    results_path,
                    palim,
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
logger.info('Really Complete.')
