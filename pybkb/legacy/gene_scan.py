import ray
from ray.util.multiprocessing import Pool
import numpy as np
import itertools
import tqdm
import compress_pickle
from collections import defaultdict
from pygobnilp.gobnilp import Gobnilp
from gurobipy import *

from pybkb.common.bn import generate_gobnilp_mdl_local_scores, BN, Node
from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.common.bayesianKnowledgeBase import HumanReadableSnode

def preprocess_features(feature_states):
    features = list(set([f for f, _ in feature_states]))
    feature_states_map = {fs: i for i, fs in enumerate(feature_states)}
    states_map = defaultdict(list)
    for f, s in feature_states:
        states_map[f].append(s)
    return features, feature_states_map, states_map

def sort_genes(data, feature_states, count_str='T'):
    variant_counts = defaultdict(int)
    for row in data:
        one_indices = np.asarray(row == 1).nonzero()[0]
        for i in one_indices:
            feature, state = feature_states[i]
            if state == count_str:
                variant_counts[feature] += 1
    return sorted([(n, f) for f, n in variant_counts.items()], reverse=True)

@ray.remote(num_cpus=1)
class Supervisor:
    def __init__(self, num_workers):
        self.num_workers = num_workers
    def work(self, genes_sets, data, feature_states, feature_states_map, states_map):
        for genes in genes_sets:
            res = get_bn(genes, data, feature_states, feature_states_map, states_map)
            if res == False:
                return res
            bn, bn_bkb, bn_mdl = res
            # Search for other bkb permutations
            bkbs = bn.make_bkb_permutations(check_mutex=False)
            pool = Pool(self.num_workers)
            pool.starmap(get_bkb_perm_mdl, [(bkb, data, feature_states) for bkb in bkbs])
            #TODO: If necessary
        return ray.get([w.work.remote() for w in self.workers])

def run_scan(genes, data, feature_states, feature_states_map, states_map, size):
    res = get_bn(genes, data, feature_states, feature_states_map, states_map, size)
    if res == False:
        return False
    bn, bn_bkb, bn_mdl = res
    return search_perms(bn, bn_bkb, data, feature_states)

def get_bkb_perm_mdl(bkb, data, feature_states):
    return bkb.calculate_mdl(data, feature_states)

def get_bn(genes, data, feature_states, feature_states_map, states_map, size):
    # Filter on genes
    filter_ids = [feature_states_map[(f,s)] for f in genes for s in states_map[f]]
    filtered_data = data[:,filter_ids]
    filtered_feature_states = [feature_states[i] for i in filter_ids]
    # Calculate local scores
    try:
        print('Calculating scores')
        scores = generate_gobnilp_mdl_local_scores(
                filtered_data,
                filtered_feature_states,
                palim=len(genes),
                inclue_no_pa=True,
                )
    except ValueError:
        # Likely that one of the genes was always/never mutated
        return False
    # Build gobnilp model
    m = Gobnilp()
    # Setup the model (we need to add a connected constraint
    m.learn(local_scores_source=scores, end='MIP model')
    # Get adjacency matrix variables
    adj = [var for pair, var in m.adjacency.items()]
    # Create constraint
    m.addConstr(sum(adj), GRB.GREATER_EQUAL, size - 1)
    # Now learn
    m.learn(local_scores_source=scores, start='MIP model')
    # Get bn learn model string to load our BN class
    modelstr = m.learned_bn.bnlearn_modelstring()
    # Make BN
    bn = BN.from_bnlearn_modelstr(modelstr, states_map)
    # Check to see if the learned bn has a node with two parents
    #found = False
    #for node in bn.nodes:
    #    if len(node.pa) >= 2:
    #        found = True
    #        break
    #if not found:
    #    return False
    # Make equivalent bkb
    bn_bkb = bn.make_bkb(no_probs=True)
    bn_mdl =  bn_bkb.calculate_mdl_ent(data, feature_states)
    return bn, bn_bkb, bn_mdl

def search_perms(bn, bn_bkb, data, feature_states):
    # Search for other bkb permutations
    world_bkbs = bn.make_all_worlds_bkbs()
    best = (-np.inf, None)
    bn_bkb_score = 0
    for bkb_world_set in itertools.product(*list(world_bkbs.values())):
        ent = 0
        total_bkb = BKB.join(bkb_world_set)
        for bkb in bkb_world_set:
            ent += bkb.calculate_mdl_ent(data, feature_states, num_inodes_override=4)
        if ent > best[0]:
            best = (ent, total_bkb)
        if total_bkb == bn_bkb:
            bn_bkb_score = ent
    best_mdl, best_bkb = best
    if best_bkb != bn_bkb:
        return best_mdl, best_bkb, bn_bkb_score, bn_bkb
    print('BKB matched BN.')
    return False

def get_highly_mutated_analysis(
        gene_variant_file,
        pasize=2,
        sort=True,
        ):
    ''' Will just sort the genes based on mutation frequency, build a bn structure
    of the most highly mutated gene as target and other genes in pa set. Then it
    will find a bkb that has a permutation with less wait.
    '''
    # Load dataset
    with open(gene_variant_file, 'rb') as f_:
        data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
    # Collect features
    features, feature_states_map, states_map = preprocess_features(feature_states)
    # Sort variants
    if sort:
        sorted_genes = sort_genes(data, feature_states)
    # Build BN
    btm_gene = sorted_genes[0][1]
    pa_genes = [g[1] for g in sorted_genes[1:(1+pasize)]]
    pa_nodes = [Node(name, states=states_map[name]) for name in pa_genes]
    node = Node(btm_gene, states_map[btm_gene], pa_nodes)
    bn = BN()
    bn.add_node(node)
    # Make BKB equiv
    bn.calculate_cpts(data, feature_states)
    bn_bkb = bn.make_bkb()
    bn_bkb_mdl = bn_bkb.calculate_mdl(data, feature_states)
    # Search perms
    bkbs = bn.make_bkb_permutations()
    mdls = []
    run_max = -np.inf
    for bkb in tqdm.tqdm(bkbs, desc='Calculating MDLs'):
        mdl = bkb.calculate_mdl(data, feature_states)
        _run_max = max(run_max, mdl)
        if _run_max != run_max:
            tqdm.tqdm.write(f'Running Max MDL: {_run_max}.')
        run_max = _run_max
        mdls.append(mdl)
    max_mdl = max(mdls)
    print(bkbs[mdls.index(max_mdl)].json())
    print(f'BN-BKB MDL: {bkb.calculate_mdl(data, feature_states)}') 
    print(f'BKB MDL: {max_mdl}') 

def scan(
        gene_variant_file,
        size=2,
        sort=True,
        top=10
        ):
    ''' Will scan the gene variant file of all gene sets of a certian size,
    learn a BKB via gobnilp, and then try to find a BKB that has a lower MDL score.
    '''
    # Load dataset
    with open(gene_variant_file, 'rb') as f_:
        data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
    # Collect features
    features, feature_states_map, states_map = preprocess_features(feature_states)
    # Sort variants
    if sort:
        sorted_genes = sort_genes(data, feature_states)[:top]
    else:
        sorted_genes = features[:top]
    # Run scan
    for gene_combo in itertools.combinations([g for _, g in sorted_genes], r=size):
        print(f'Checking {gene_combo}')
        res = run_scan(gene_combo, data, feature_states, feature_states_map, states_map, size)
        if res:
            return res
    return False

if __name__ == '__main__':
    res = scan('../python_base/learning/variant_data-all.cpk')
    if res:
        best_bkb_score, best_bkb, bn_bkb_score, bn_bkb = res
        print(best_bkb.json())
        print(best_bkb_score)
        print(bn_bkb_score)

    # Load dataset
    with open('../python_base/learning/variant_data-all.cpk', 'rb') as f_:
        data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
    # Collect features
    features, feature_states_map, states_map = preprocess_features(feature_states)

    print('BN BKB MDLs')
    for snode in bn_bkb._S_nodes:
        print(HumanReadableSnode.make(snode, bn_bkb))
        print(snode.calculate_mdl_ent(bn_bkb, 1, data, feature_states, feature_states_map, {}, only_mdl='data'))
    print('BKB MDLs')
    for snode in best_bkb._S_nodes:
        print(HumanReadableSnode.make(snode, best_bkb))
        print(snode.calculate_mdl_ent(bn_bkb, 1, data, feature_states, feature_states_map, {}, only_mdl='data'))
