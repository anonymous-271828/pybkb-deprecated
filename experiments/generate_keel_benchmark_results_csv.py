import os
import json
import csv
import compress_pickle
import tqdm

# Get common palims for bkbs and bns from saved file
# We already saved these results in a file
with open('scores/palim_common_options.json', 'r') as f_:
    common_options = json.load(f_)

rows = [
        ['dataset', 'palim', 'num_features', 'num_inodes', 'num_examples', 'bkb_data_score', 'bn_data_score', 'score_diff', 'bkb_ncalls', 'bn_ncalls', 'ncalls_diff']
        ]

# Go through the results directory and get the reports
for dataset_name in tqdm.tqdm(os.listdir('results/bkb')):
    if dataset_name == 'tcga':
        continue
    # Get ran palim
    palim = common_options[dataset_name]
    # Load bkb report
    with open(os.path.join('results/bkb', dataset_name, 'report.json'), 'r') as f_:
        bkb_report = json.load(f_)
    # Load bn report
    with open(os.path.join('results/bn', dataset_name, 'report.json'), 'r') as f_:
        bn_report = json.load(f_)
    # Get the store lens for bkb and bn score calcs
    with open(os.path.join('scores/bkb', dataset_name, f'palim-{palim}.store'), 'rb') as f_:
        bkb_store = compress_pickle.load(f_, compression='lz4')
    with open(os.path.join('scores/bn', dataset_name, f'palim-{palim}.store'), 'rb') as f_:
        bn_store = compress_pickle.load(f_, compression='lz4')
    # Get data 
    with open(os.path.join('../data/keel/bkb_data', f'{dataset_name}.dat'), 'rb') as f_:
        data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
    features = set([f for f, s in feature_states])
    bkb_score = bkb_report["scores"]["data"]
    bn_score = bn_report["like bkb scores"]["data"]
    # Subtract 2 because we have two keys that are used for reporting and are not actually probabilities
    bkb_ncalls = len(bkb_store) - 2
    bn_ncalls = len(bn_store) - 2
    # Make row
    row = [
            dataset_name, 
            palim,
            len(features),
            len(feature_states),
            data.shape[0],
            bkb_score,
            bn_score,
            bkb_score - bn_score,
            bkb_ncalls,
            bn_ncalls,
            bkb_ncalls - bn_ncalls,
            ]
    rows.append(row)

# Write csv file
with open('mdl_analysis.csv', 'w') as f_:
    writer = csv.writer(f_)
    writer.writerows(rows)
