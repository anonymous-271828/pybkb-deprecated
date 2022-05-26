import os
import json
import ray
import logging
import compress_pickle
import argparse
from operator import itemgetter

from pybkb.learn import BKBLearner, BNLearner
from pybkb.utils.data import KeelWrangler
from pybkb.utils.mp import MPLogger

# Collect highest palim that are common between both bn and bkb score runs
# We already saved these results in a file, maybe create your own cause your max palim might be different.
#with open('scores/palim_common_options.json', 'r') as f_:
#    common_options = json.load(f_)

# Now let's sort the datasets based on feature states length as thats similar enough to data len
# We've also saved these ordering in a file, maybe create your own cause your max palim might be different.
#with open('../data/keel/feature_states_order.json', 'r') as f_:
#    feature_states_order = json.load(f_)

def main(datasets_path, scores_path, results_path, palim, dataset_name=None):
    if dataset_name is None:
        datasets = os.listdir(os.path.join(scores_path, 'bkb'))
    else:
        datasets = [dataset_name]
    # Initialize ray cluster
    ray.init(address='auto')
    # Initialize logger
    logger = MPLogger('Main', logging.INFO, loop_report_time=60)
    # Learn BKBs and BNs
    for i, dataset_name in enumerate(datasets):
        logger.info(f'Learning on {dataset_name} with palim = {palim}. {i}/{len(datasets)} to go.')
        # Load dataset
        dataset = f'{dataset_name}.dat'
        datasets_path = os.path.join(datasets_path, dataset)
        with open(datasets_path, 'rb') as f_:
            data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
        ## Learn BKB from scores
        result_path_bkb = os.path.join(results_path, 'bkb', dataset_name)
        # Get scores and store paths for bkb
        scores_bkb_path = os.path.join(scores_path, 'bkb', dataset_name, f'palim-{palim}.scores')
        store_bkb_path = os.path.join(scores_path, 'bkb', dataset_name, f'palim-{palim}.store')
        with open(scores_bkb_path, 'rb') as f_:
            scores = compress_pickle.load(f_, compression='lz4')
        with open(store_bkb_path, 'rb') as f_:
            store = compress_pickle.load(f_, compression='lz4')
        logger.info('Store and scores loaded')
        # Initialize learner
        bkb_learner = BKBLearner('gobnilp', 'mdl_ent', palim=palim, distributed=False)
        # Fit
        logger.info('Fitting BKB...')
        bkb_learner.fit(data, feature_states, srcs, scores=scores, store=store)
        if not os.path.exists(result_path_bkb):
            os.makedirs(result_path_bkb)
        logger.info('Completed BKB learning and now saving.')
        # Save learned bkb
        bkb_learner.learned_bkb.save(os.path.join(result_path_bkb, 'learned.bkb'))
        # Save report
        bkb_learner.report.json(filepath=os.path.join(result_path_bkb, 'report.json'))   
        ## Learn BN from scores
        result_path_bn = os.path.join(results_path, 'bn', dataset_name)
        # Get scores and store paths for bn
        logger.info('Loading BN scores and store.')
        scores_bn_path = os.path.join(scores_path, 'bn', dataset_name, f'palim-{palim}.scores')
        store_bn_path = os.path.join(scores_path, 'bn', dataset_name, f'palim-{palim}.store')
        with open(scores_bn_path, 'rb') as f_:
            scores = compress_pickle.load(f_, compression='lz4')
        with open(store_bn_path, 'rb') as f_:
            store = compress_pickle.load(f_, compression='lz4')
        # Initialize learner
        bn_learner = BNLearner('gobnilp', 'mdl_ent', palim=palim)
        # Fit
        logger.info('Fitting BN...')
        bn_learner.fit(data, feature_states, srcs, scores=scores, store=store)
        if not os.path.exists(result_path_bn):
            os.makedirs(result_path_bn)
        logger.info('Completed BN learning and now saving.')
        # Save learned bn
        with open(os.path.join(result_path_bn, 'learned.bn'), 'wb') as f_:
            compress_pickle.dump(bn_learner.bn, f_, compression='lz4')
        # Save report
        bn_learner.report.json(filepath=os.path.join(result_path_bn, 'report.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Master path to all preprocessed BKB datasets.')
    parser.add_argument('results_path', help='Path to directory where results should be saved.')
    parser.add_argument('scores_path', help='Path to directory where scores have been saved.')
    parser.add_argument('palim', help='Parent set Limit that of the scores file you wish to run learning on.',default=2)
    parser.add_argument('--dataset_name', help='Name of dataset to run.', default=None)
    args = parser.parse_args()

    # Run main script
    main(
            args.dataset_path,
            args.scores_path,
            args.results_path,
            args.palim,
            args.dataset_name,
            )
