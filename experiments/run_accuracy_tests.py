import os
import json
import csv
import compress_pickle
import tqdm
import ray
import copy
from ray.util.placement_group import placement_group
import logging

from collections import defaultdict

from pybkb.bkb import BKB
from pybkb.bn import BN
from pybkb.reason import BaseReasoner
from pybkb.reason.bkb import BKBReasoner
from pybkb.reason.bn import BNReasoner
from pybkb.utils.mp import MPLogger

RESULTS_BKB_PATH = 'results/cross_validation/bkb'
RESULTS_BN_PATH = 'results/cross_validation/bn'
DATASET_PATH = '../data/keel/bkb_data/cross_validation_sets'

# Load class variable map file
with open('../data/keel/class_variable_map.json', 'r') as f_:
    class_vars = json.load(f_)

NUM_NODES = 15
WORKERS_PER_NODE = 17

## Functions
def setup_cluster():
    # Initialize Ray
    ray.init(address='auto')
    # Setup placement group and bundles
    bundles = [{"CPU": WORKERS_PER_NODE} for _ in range(NUM_NODES)]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    return pg, bundles

@ray.remote(num_cpus=1)
def get_cross_predictions(
        dataset_name,
        dataset_path,
        results_bkb_path,
        results_bn_path,
        crossvalid_file,
        crossvalid_idx,
        class_vars,
        ):
    logger = MPLogger(f'{dataset_name} Worker', logging.INFO, id=crossvalid_idx, loop_report_time=60)
    crossvalid_name = crossvalid_file.split('.')[0]
    # Load cross validation dataset
    with open(dataset_path, 'rb') as f_:
        data_train, data_test, feature_states, _, srcs_test = compress_pickle.load(f_, compression='lz4')
    # Makes states for BN
    states = defaultdict(list)
    for feature, state in feature_states:
        states[feature].append(state)
    states = dict(states)
    # Load bkb
    bkb_path = os.path.join(results_bkb_path, crossvalid_file)
    bkb = BKB.load(bkb_path)
    # Load BN
    bn_path = os.path.join(results_bn_path, f'{crossvalid_name}.bn')
    with open(bn_path, 'rb') as f_:
        # This is PyGobnilp BN obj
        bn = compress_pickle.load(f_, compression='lz4')
    # Load our BN obj
    bn = BN.from_bnlearn_modelstr(bn.bnlearn_modelstring(), states)
    # Calculate probs from training data
    bn.calculate_cpts_from_data(data_train, feature_states)
    # Get BKB predictions
    bkb_reasoner = BKBReasoner(bkb=bkb)
    logger.info('Starting BKB Reasoning...')
    logger.initialize_loop_reporting()
    bkb_preds, bkb_truths = bkb_reasoner.predict(
            class_vars[dataset_name],
            data_test,
            feature_states,
            collect_truths=True, 
            heuristic='fused_with_complete',
            verbose=0,
            logger=logger,
            )
    logger.info('Starting BN Reasoning...')
    # Get BN predictions
    bn_reasoner = BNReasoner(bn)
    bn_preds, bn_truths = bn_reasoner.predict(class_vars[dataset_name], data_test, feature_states, collect_truths=True)
    logger.info('Finished')
    return (bkb_preds, bkb_truths, bn_preds, bn_truths, dataset_name)

def add_acc_results_to_file(dataset_name, dataset_res):
    # Collect predictions
    # Combine cross valids
    bkb_predictions_total = []
    bkb_truths_total = []
    for cross_bkb_predictions, cross_bkb_truths in zip(dataset_res["bkb_predictions"], dataset_res["bkb_truths"]):
        bkb_predictions_total.extend(list(cross_bkb_predictions))
        bkb_truths_total.extend(list(cross_bkb_truths))
    bn_predictions_total = []
    bn_truths_total = []
    for cross_bn_predictions, cross_bn_truths in zip(dataset_res["bn_predictions"], dataset_res["bn_truths"]):
        bn_predictions_total.extend(list(cross_bn_predictions))
        bn_truths_total.extend(list(cross_bn_truths))
    bkb_acc = BaseReasoner.precision_recall_fscore_support(bkb_truths_total, bkb_predictions_total, average='weighted')
    bn_acc = BaseReasoner.precision_recall_fscore_support(bn_truths_total, bn_predictions_total, average='weighted')

    row = [
            dataset_name,
            bkb_acc[0][0],
            bkb_acc[0][1],
            bkb_acc[0][2],
            len(bkb_acc[2]),
            bn_acc[0][0],
            bn_acc[0][1],
            bn_acc[0][2],
            ]

    # Write csv file
    with open('acc_analysis.csv', 'a') as f_:
        writer = csv.writer(f_)
        writer.writerow(row)
# Setup logging
logger = MPLogger('Main', logging.INFO, loop_report_time=60)

# Make into nice csv file with header
row = ['dataset', 'bkb_precision', 'bkb_recall', 'bkb_f1', 'num_no_inf', 'bn_precision', 'bn_recall', 'bn_f1']
with open('acc_analysis.csv', 'w') as f_:
    writer = csv.writer(f_)
    writer.writerow(row)

# Setup Cluster
pg, bundles = setup_cluster()

# Setup score collection work
acc_ids = []
bundle_idx = 0
dataset_counter = 0
for dataset_name in os.listdir('results/cross_validation/bkb'):
    results_bkb_path = os.path.abspath(os.path.join(RESULTS_BKB_PATH, dataset_name))
    results_bn_path = os.path.abspath(os.path.join(RESULTS_BN_PATH, dataset_name))
    crossvalid_files = [fn for fn in os.listdir(results_bkb_path) if '.bkb' in fn]
    if len(crossvalid_files) < 10:
        continue
    for idx, crossvalid_file in enumerate(crossvalid_files):
        dataset_counter += 1
        if dataset_counter > WORKERS_PER_NODE:
            # Move on to the next bundle
            dataset_counter = 0
            bundle_idx += 1
        crossvalid_name = crossvalid_file.split('.')[0]
        # Load cross validation dataset
        dataset_path = os.path.abspath(os.path.join(DATASET_PATH, dataset_name, f'{crossvalid_name}.dat'))
        acc_ids.append(
                get_cross_predictions.options(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx
                    ).remote(
                            dataset_name,
                            dataset_path,
                            results_bkb_path,
                            results_bn_path,
                            crossvalid_file,
                            idx,
                            class_vars,
                        )
                    )

i = 0
total = len(acc_ids)
dict_template = {
            "bkb_predictions": [],
            "bkb_truths": [],
            "bn_predictions": [],
            "bn_truths": [],
            }
res = defaultdict(lambda: copy.deepcopy(dict_template))
logger.initialize_loop_reporting()
while len(acc_ids):
    done_ids, acc_ids = ray.wait(acc_ids)
    bkb_preds, bkb_truths, bn_preds, bn_truths, dataset_name = ray.get(done_ids[0])
    res[dataset_name]["bkb_predictions"].append(bkb_preds)
    res[dataset_name]["bkb_truths"].append(bkb_truths)
    res[dataset_name]["bn_predictions"].append(bn_preds)
    res[dataset_name]["bn_truths"].append(bn_truths)
    if len(res[dataset_name]["bkb_predictions"]) == 10:
        add_acc_results_to_file(dataset_name, res[dataset_name])
    i += 1
    logger.report(i=i, total=total)

