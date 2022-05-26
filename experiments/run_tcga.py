import os
import compress_pickle
import logging
import argparse

from pybkb.learn import BKBLearner
from pybkb.utils.mp import MPLogger


def main(dataset_path, results_path):
    logger = MPLogger('Main', logging.INFO)

    with open(dataset_path, 'rb') as f_:
        data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')

    learner = BKBLearner('gobnilp', 'mdl_ent', palim=1, distributed=False)
    learner.fit(data, feature_states, srcs)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Save BKB
    learner.learned_bkb.save(os.path.join(results_path, 'tcga-brca-2.bkb'))
    # Save report
    learner.report.json(filepath=os.path.join(results_path, 'tcga-brca-2-report.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Master path to all preprocessed BKB datasets.')
    parser.add_argument('results_path', help='Path to directory where results should be saved.')
    args = parser.parse_args()

    # Run main script
    main(
            args.dataset_path,
            args.results_path,
            )
