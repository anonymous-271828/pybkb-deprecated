import os
import json
import ray
import logging
import compress_pickle
from operator import itemgetter

from pybkb.learn import BKBLearner, BNLearner
from pybkb.utils.data import KeelWrangler
from pybkb.utils.mp import MPLogger


# Collect highest palim that are common between both bn and bkb score runs
# We already saved these results in a file, or edit or make your own
with open('palim_common_options.json', 'r') as f_:
    common_options = json.load(f_)

# Now let's sort the datasets based on feature states length as thats similar enough to data len
# We've also saved these ordering in a file, or make your own
with open('feature_states_order.json', 'r') as f_:
    feature_states_order = json.load(f_)

# Initialize ray cluster
#ray.init(address='auto')

# Initialize logger
logger = MPLogger('Main', logging.INFO, loop_report_time=60)

# Learn BKBs and BNs based on feature states ordering, i.e. easiest to hardest
for i, (_, dataset) in enumerate(feature_states_order):
    dataset_name = dataset.split('.')[0]
    # Get the max palim that is common to both bn and bkb
    try:
        palim = common_options[dataset_name]
    except KeyError:
        continue
    ## Learn BKB from scores
    result_path_bkb = os.path.join('results/cross_validation/bkb', dataset_name)
    # Get scores and store paths for bkb
    scores_bkb_path = os.path.join('scores/cross_validation/bkb', dataset_name)
    cross_valid_scores = os.listdir(scores_bkb_path)
    # Get scores and store paths for bn
    if len(cross_valid_scores) < 10:
        continue
    ## Learn BN from scores
    result_path_bn = os.path.join('results/cross_validation/bn', dataset_name)
    scores_bn_path = os.path.join('scores/cross_validation/bn', dataset_name)
    logger.info(f'Learning on {dataset_name} with palim = {palim}. {i}/{len(feature_states_order)} to go.')
    # Load dataset
    datasets_path = os.path.join('../data/keel/bkb_data/cross_validation_sets', dataset_name)
    for cross_valid_score_file in cross_valid_scores:
        cross_valid_name = cross_valid_score_file.split('.')[0]
        with open(os.path.join(datasets_path, f'{cross_valid_name}.dat'), 'rb') as f_:
            data, _, feature_states, srcs, _ = compress_pickle.load(f_, compression='lz4')
        crossvalid_scores_bkb_path = os.path.join(scores_bkb_path, cross_valid_score_file)
        with open(crossvalid_scores_bkb_path, 'rb') as f_:
            scores = compress_pickle.load(f_, compression='lz4')
        logger.info('Scores loaded')
        # Initialize learner
        bkb_learner = BKBLearner('gobnilp', 'mdl_ent', palim=palim, distributed=False)
        # Fit
        logger.info('Fitting BKB...')
        bkb_learner.fit(data, feature_states, srcs, scores=scores)
        if not os.path.exists(result_path_bkb):
            os.makedirs(result_path_bkb)
        logger.info('Completed BKB learning and now saving.')
        # Save learned bkb
        bkb_learner.learned_bkb.save(os.path.join(result_path_bkb, f'{cross_valid_name}.bkb'))
        # Save report
        bkb_learner.report.json(filepath=os.path.join(result_path_bkb, f'{cross_valid_name}-report.json'))   
        logger.info('Loading BN scores.')
        crossvalid_scores_bn_path = os.path.join(scores_bn_path, cross_valid_score_file)
        with open(crossvalid_scores_bn_path, 'rb') as f_:
            scores = compress_pickle.load(f_, compression='lz4')
        # Initialize learner
        bn_learner = BNLearner('gobnilp', 'mdl_ent', palim=palim)
        # Fit
        logger.info('Fitting BN...')
        bn_learner.fit(data, feature_states, srcs, scores=scores)
        if not os.path.exists(result_path_bn):
            os.makedirs(result_path_bn)
        logger.info('Completed BN learning and now saving.')
        # Save learned bn
        with open(os.path.join(result_path_bn, f'{cross_valid_name}.bn'), 'wb') as f_:
            compress_pickle.dump(bn_learner.bn, f_, compression='lz4')
        # Save report
        bn_learner.report.json(filepath=os.path.join(result_path_bn, f'{cross_valid_name}-report.json'))
