from collections import defaultdict
import itertools
import numpy as np
import pandas as pd

def joint_prob(data, x_state_idx=None, parent_state_indices=None, store=None):
    if store is None:
        store = {}
    if x_state_idx is None:
        x_state_idx = []
    else:
        x_state_idx = [x_state_idx]
    if parent_state_indices is None:
        parent_state_indices = []
    indices = tuple(sorted(x_state_idx + parent_state_indices))
    if indices in store:
        prob = store[indices]
    else:
        count = 0
        for row in data:
            truth = True
            for s_idx in indices:
                if row[s_idx] != 1:
                    truth = False
                    break
            if truth:
                count += 1
        prob = count / len(data)
        store[indices] = prob
    return prob, store

def instantiated_entropy(data, x_state_idx, parent_state_indices, store=None):
    p_xp, store = joint_prob(data, x_state_idx, parent_state_indices, store)
    p_p, store = joint_prob(data, x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    if p_xp == 0:
        return 0, store
    try:
        ent = p_xp * np.log2(p_xp / p_p)
    except MathDomainError:
        return 0, store
    return ent, store

def instantiated_mutual_info(data, x_state_idx, parent_state_indices, store=None):
    p_xp, store = joint_prob(data, x_state_idx, parent_state_indices, store)
    p_x, store = joint_prob(data, x_state_idx=x_state_idx, parent_state_indices=None, store=store)
    p_p, store = joint_prob(data, x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    if p_xp == 0:
        return 0, store
    try:
        mi = p_xp * np.log2(p_xp / (p_x * p_p))
    except MathDomainError:
        mi = 0
    return mi, store

def instantiated_ll(data, x_state_idx, parent_state_indices, store=None):
    p_xp, store = joint_prob(data, x_state_idx, parent_state_indices, store)
    p_p, store = joint_prob(data, x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    if p_xp == 0:
        return 0, store
    try:
        ll = p_xp * np.log2(p_xp / p_p)
    except MathDomainError:
        ll = 0
    return ll, store

def build_feature_state_map(feature_states):
    feature_states_map = defaultdict(list)
    for idx, (feature, state) in enumerate(feature_states):
        feature_states_map[feature].append((idx, state))
    return feature_states_map

def format_imi_table(table, feature_states):
    temp_list = []
    for row in table:
        temp_row = []
        for i in range(len(row) - 1):
            temp_row.append(feature_states[row[i]])
        temp_row.append(row[-1])
        temp_list.append(temp_row)
    temp_dict = defaultdict(list)
    for row in temp_list:
        for i in range(len(row) - 1):
            temp_dict[row[i][0]].append(row[i][1])
        temp_dict["Mutual Info"].append(row[-1])
    return pd.DataFrame.from_dict(temp_dict)

def entropy(data, x, feature_states, feature_states_map, parents=None, store=None):
    if store is None:
        store = {}
    if parents is None:
        parents = []
    ent = 0
    parent_state_sets = [[s_idx for s_idx, _ in feature_states_map[p]] for p in parents]
    for x_state_idx, _ in feature_states_map[x]:
        if len(parents) == 0:
            _ent, store = instantiated_entropy(data, x_state_idx, parents, store)
            ent += _ent
            continue
        for parent_state_indices in itertools.product(*parent_state_sets):
            _ent, store = instantiated_entropy(data, x_state_idx, list(parent_state_indices), store)
            ent += _ent
    return -ent, store

def mutual_info(data, x, parents, feature_states, feature_states_map, store=None):
    if store is None:
        store = {}
    imi_table = []
    mi = 0
    parent_state_sets = [[s_idx for s_idx, _ in feature_states_map[p]] for p in parents]
    for x_state_idx, _ in feature_states_map[x]:
        for parent_state_indices in itertools.product(*parent_state_sets):
            imi_row = [x_state_idx] + list(parent_state_indices)
            _mi, store = instantiated_mutual_info(data, x_state_idx, list(parent_state_indices), store)
            mi += _mi
            imi_row.append(_mi)
            imi_table.append(imi_row)
    return mi, format_imi_table(imi_table, feature_states), store

def logl(data, x, parents, feature_states, feature_states_map, store=None):
    if store is None:
        store = {}
    ill_table = []
    ll = 0
    parent_state_sets = [[s_idx for s_idx, _ in feature_states_map[p]] for p in parents]
    for x_state_idx, _ in feature_states_map[x]:
        for parent_state_indices in itertools.product(*parent_state_sets):
            ill_row = [x_state_idx] + list(parent_state_indices)
            _ll, store = instantiated_ll(data, x_state_idx, list(parent_state_indices), store)
            ll += _ll
            ill_row.append(_ll)
            ill_table.append(ill_row)
    return ll, format_imi_table(ill_table, feature_states), store

def load_sprinkler():
    import pickle
    with open('sprinkler.dat', 'rb') as f_:
        data, feature_states, srcs = pickle.load(f_)

    feature_states_map = build_feature_state_map(feature_states)
    return data, feature_states, feature_states_map

def sprinkler():
    data, feature_states, feature_states_map = load_sprinkler()
    w, iw, store = mutual_info(data, 'cloudy', ['rain'], feature_states, feature_states_map)
    print(feature_states)
    print(w)
    print(iw)
    print(store)

#sprinkler()
