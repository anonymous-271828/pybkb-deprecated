import time
import tqdm
import os
import itertools
import math
import random
import pickle
import logging
from collections import defaultdict
import json
import numpy as np
import logging
import sys

from pybkb.python_base.utils import get_operator, try_convert_to, get_opposite_operator

logger = logging.getLogger(__name__)

def recursive_dd(max_depth, end_type, cur_depth=0):
    if cur_depth < max_depth:
        return defaultdict(recursive_dd(max_depth, end_type, cur_depth=cur_depth+1))
    else:
        return defaultdict(end_type)

class JointReasoner:
    def __init__(
            self,
            dataset=None,
            discretize=None,
            gene_interpolations=None,
            drug_interpolations=None
            ):
        """ Joint Reasoner is responsible for taking in a dataset dictionary and calculating joint probabilities
            explicitly. No BKBs required. Should be used to access a BKBs joint distribution against a dataset's
            ground truth joint distribution.

            :param patient_data: A data dictionary containing each exemplar's features and associated levels,
                values or states. It should be of the form:
                {
                    example_1:
                        {
                            feature_1: value_feature_1,
                            feature_2: value_feature_2,
                            ...
                            feature_m: value_feature_m
                        },
                    ...
                    example_n: ...
                }
            :type dataset: dict
        """

        self.data_dict = dataset
        if dataset is not None:
            self.set_dataset(dataset, discretize)
        if gene_interpolations is not None:
            self.gene_interpolations = gene_interpolations
        if drug_interpolations is not None:
            self.drug_interpolations = drug_interpolations
        self.interp_type = None

    def _setup(self):
        start_time = time.time()
        logger.info('Starting setup.')
        self.num_examples = len(self.data_dict)
        self.features = self._collect_features()
        self.features_list = list(self.features.keys())
        self.states = [[state for state in self.features[feature]] for feature in self.features_list]
        self.unique_feature_set_counts, self.unique_feature_set_examples = self._calculate_unique_feature_set_counts()
        self.feature_map = {idx: feature_name for idx, feature_name in enumerate(self.features)}
        logger.info('Completed setup in {} seconds.'.format(time.time() - start_time))

    def set_dataset(self, dataset, discretize=None):
        """ Sets up data dict. Might want to add support for pandas or numpy later.
        """
        # Discretize data dictionary if necessary.
        if discretize is None:
            self.data_dict = dataset
        else:
            self.data_dict, self.levels = self._discretize(dataset, discretize)
        self._setup()

    def _collect_features(self):
        """ Collects a dictionary of all features and their respective levels/states.

        :return: A dictionary of {features: [levels]}.
        :rtype: dict
        """
        _features = defaultdict(set)
        for example, feature_dict in self.data_dict.items():
            for feature, state in feature_dict.items():
                if type(state) == list:
                    for _state in state:
                        _features[feature].add(_state)
                else:
                    _features[feature].add(state)
        features = {}
        for feature, states in _features.items():
            features[feature] = list(states)
        return features

    def _calculate_unique_feature_set_counts(self):
        """ This function will count how many times each unique feature set, i.e. a unquie combination of feature levels,
            appeared in the data.

        :return: The counts in the form { (f1_state, f2_state, ...): {count} }.
        :rtype: dict

        Notice: feature ordering is determined by features_list initialization.
        """
        counts = defaultdict(int)
        examples_dict = defaultdict(list)
        for example, feature_dict in self.data_dict.items():
            feature_set = []
            for feature in self.features_list:
                if type(feature_dict[feature]) == list:
                    feature_set.append(tuple(sorted(feature_dict[feature])))
                else:
                    feature_set.append(feature_dict[feature])
            counts[tuple(feature_set)] += 1
            examples_dict[tuple(feature_set)].append(example)
        return counts, examples_dict

    def _are_consistent_feature_sets(self, feature_set_1, feature_set_2):
        """ Returns whether two feature sets are consistent, i.e. whether they are equal or contained by each other.
        """
        try:
            for feature_state_1, feature_state_2 in zip(feature_set_1, feature_set_2):
                if feature_state_1 != feature_state_2:
                    # Check if either feature state is a co-occuring feature state
                    if type(feature_state_1) == list or type(feature_state_2) == list:
                        # Convert feafure states to sets to perform check
                        if type(feature_state_1) == list:
                            feature_state_1_set = set(feature_state_1)
                        else:
                            feature_state_1_set = {feature_state_1}
                        if type(feature_state_2) == list:
                            feature_state_2_set = set(feature_state_2)
                        else:
                            feature_state_2_set = {feature_state_2}
                        # Perform check
                        min_set = min(len(feature_state_1), len(feature_state_2))
                        if len(feature_state_1_set & feature_state_2_set) != min_set:
                            raise InvalidFeatureState
                    if feature_state_1 is not None and feature_state_2 is not None:
                        raise InvalidFeatureState
            return True
        except InvalidFeatureState:
            return False

    def _calculate_conditional_prob(self, A, B=None):
        """ Calculate the conditional probability P(A|B) = P(A,B) / P(B) from the dataset.

        :param A: A tuple of target (feature_name, state_name).
        :type A: tuple
        :param B: A evidence dictionary of the form [(feature_name, state_name), ...] or if
            co-occuring states, has the form [(feature_name, [state_name_1, state_name_2, ...]), ...]
        :type B: list

        :return: Conditional probability
        :rtype: float
        """
        if B is None:
            B = []
        # Initialize
        count_ab = 0
        count_b = 0
        feature_set_ab = [None for _ in  range(len(self.features))]
        feature_set_b = [None for _ in range(len(self.features))]

        # Build feature set for the joint A and B.
        AB = B + [A]
        for feature_name, state_name in AB:
            feature_idx = self.features_list.index(feature_name)
            feature_set_ab[feature_idx] = state_name

        # Build feature set for just B.
        for feature_name, state_name in B:
            feature_idx = self.features_list.index(feature_name)
            feature_set_b[feature_idx] = state_name

        # Count number of dataset feature sets that are consistent with AB and B.
        for feature_set, count in self.unique_feature_set_counts.items():
            if self._are_consistent_feature_sets(feature_set, feature_set_ab):
                count_ab += count
            if self._are_consistent_feature_sets(feature_set, feature_set_b):
                count_b += count

        prob = float(count_ab / count_b) if count_b > 0 else -1
        return prob

    def compute_joint(self,
                      evidence=None,
                      targets=None,
                      continuous_evidence=None,
                      continuous_targets=None,
                      contribution_features=None,
                      interpolation_type=None):
        """ Computes the joint probability of the targets and the evidence, i.e. P(A,B).

            :param evidence: A dictionary of evidence of the form:
                {
                    feature_1: value_feature_1,
                    feature_2: value_feature_2,
                    ...,
                    feature_k: value_feature_k
                }
            :type evidence: dict
            :param targets: A list of features to update, i.e. calculate joint with evidence.
            :type targets: list
            :param evidence: A dictionary of continous evidence of the form:
                {
                    feature_1: {
                        "op": { One of '>=', '<=', '==', 'in' },
                        "value": { One of int, float, list }
                    },
                    ...
                }
            :type evidence: dict
            :param targets: A dict of continuous targets to update of the same form as continuous evidence., i.e. calculate joint with evidence.
            :type targets: dict
            :param contribution_features: A list of features to calculate contributions.
            :type contribution_features: list
            :param discretize: The number of levels to discretize continous variables.
            :param discretize: int, dict
        """
        self.interp_type = interpolation_type
        # Convert evidence to list indices for future feature set calculations
        processed_evidence = {}
        # Process categorical evidence
        if evidence is not None:
            for feature, state in evidence.items():
                feature_idx = self.features_list.index(feature)
                processed_evidence[feature_idx] = state
        # Process continuous evidence
        if continuous_evidence is not None:
            for feature, prop in continuous_evidence.items():
                feature_idx = self.features_list.index(feature)
                processed_evidence[feature_idx] = prop
        logger.info('Processed evidence.')
        processed_targets = {}
        # Process categorical targets
        if targets is not None:
            for target_feature in targets:
                feature_idx = self.features_list.index(target_feature)
                processed_targets[feature_idx] = None
        # Process continuous targets
        if continuous_targets is not None:
            for feature, prop in continuous_targets.items():
                # Add in specified property
                feature_idx = self.features_list.index(feature)
                processed_targets[feature_idx] = [prop]
                # Add in opposite of specified property
                processed_targets[feature_idx].append({
                    "op": get_opposite_operator(prop["op"]),
                    "value": prop["value"]
                })
        logger.info('Processed targets.')
        if contribution_features is not None:
            non_evidence_feature_indices = self._get_contribution_indices(
                contribution_features,
                processed_evidence,
                processed_targets,
            )
            logger.info('Processed contribution variables.')
        # Setup for reasoning
        contributions_table = defaultdict(lambda: defaultdict(int))
        results = defaultdict(int)
        evidence_inferences = 0
        logger.info('Starting reasoning.')
        start_time = time.time()
        traversed_examples = set()
        # Calculate updates
        for (feature_set, count), (_, examples) in tqdm.tqdm(
            zip(self.unique_feature_set_counts.items(),
                self.unique_feature_set_examples.items()),
            desc='Finding supported inferences',
            total=len(self.unique_feature_set_counts),
            leave=False):
            is_consistent, interp_probs = self._is_consistent_with_evidence(processed_evidence, feature_set)
            if is_consistent:
                evidence_inferences += self._prob(count)
                for target, target_prop in processed_targets.items():
                    if target_prop is None:
                        for state in self.states[target]:
                            # interp prob is always 1 unless we use interpolation
                            if self._is_consistent_with_target_state(target, state, feature_set):
                                # Don't double count examples
                                num_examples_already_counted = len(set(examples) - traversed_examples)
                                if num_examples_already_counted != count:
                                    count -= num_patients_already_counted
                                    logger.info('Already encountered some patients.')
                                inference_prob = self._prob(count)
                                for interp_prob in interp_probs:
                                    inference_prob *= interp_prob
                                results[(self.feature_map[target], state)] += inference_prob
                                if contribution_features is not None:
                                    contributions_table = self._update_contributions(
                                        contributions_table,
                                        target,
                                        state,
                                        feature_set,
                                        non_evidence_feature_indices,
                                        count,
                                        interp_probs
                                    )
                    else:
                        for prop in target_prop:
                            if self._is_consistent_with_target_state(target, prop, feature_set):
                                # Don't double count examples
                                num_examples_already_counted = len(set(examples) - traversed_examples)
                                if num_examples_already_counted != count:
                                    count -= num_patients_already_counted
                                    logger.info('Already encountered some patients.')
                                inference_prob = self._prob(count)
                                for interp_prob in interp_probs:
                                    inference_prob *= interp_prob
                                results[(self.feature_map[target], '{} {}'.format(prop["op"], prop["value"]))] += inference_prob
                                if contribution_features is not None:
                                    state = '{} {}'.format(prop['op'], prop['value'])
                                    contributions_table = self._update_contributions(
                                        contributions_table,
                                        target,
                                        state,
                                        feature_set,
                                        non_evidence_feature_indices,
                                        count,
                                        interp_probs
                                    )
        logger.info('Finished reasoning in {} seconds.'.format(time.time() - start_time))
        return dict(results), contributions_table

    def _get_contribution_indices(self, contribution_features, processed_evidence, processed_targets):
        if contribution_features == 'all':
            return list(set([feature_idx for feature_idx in range(len(self.features))]) - set(list(processed_evidence.keys()) + list(processed_targets.keys())))
        elif contribution_features == 'ENSEMBL':
            keep_indices = set()
            for idx, feature in enumerate(self.features.keys()):
                if 'ENSEMBL' in feature:
                    keep_indices.add(idx)
            return list(keep_indices)
        elif contribution_features == 'CHEMBL.COMPOUND':
            keep_indices = set()
            for idx, feature in enumerate(self.features.keys()):
                if 'CHEMBL.COMPOUND' in feature:
                    keep_indices.add(idx)
            return list(keep_indices)
        else:
            return []
            #return [self.features_list.index(feature_name) for feature_name in contribution_features]

    def _prob(self, count):
        return float(count / self.num_examples)

    def _update_contributions(self, contributions_table, target, state, feature_set, non_evidence_feature_indices, count, interp_probs):
        for non_evidence_feature in non_evidence_feature_indices:
            target_name = self.feature_map[target]
            non_evidence_name = self.feature_map[non_evidence_feature]
            contribution_prob = self._prob(count)
            for interp_prob in interp_probs:
                contribution_prob *= interp_prob
            contributions_table[(target_name, state)][(non_evidence_name, feature_set[non_evidence_feature])] += contribution_prob
        return contributions_table

    def _is_consistent_with_evidence(self, evidence, feature_set):
        # is a list containing only 1. If we use interpolation we append that interpolation probability to be multiplied into the inference later.
        interp_probability = [1]
        for feature, state_or_prop in evidence.items():
            if type(state_or_prop) == dict:
                if (self.interp_type == 'gene' and self.gene_interpolations is not None) or (self.interp_type == 'drug' and self.drug_interpolations is not None):
                    op = get_operator(state_or_prop["op"])
                    value = state_or_prop["value"]
                    feature_set_state = try_convert_to(feature_set[feature], type(value))
                    if not op(feature_set_state, value):
                        # check bigram
                        if 'ENSEMBL' in self.feature_map[feature]:
                            base_node = (self.feature_map[feature], op, value)
                            interp_node = self.gene_interpolations[base_node]['interpolation']
                            interp_node_idx = self.features_list.index(interp_node[0])
                            interp_feature_set_state = try_convert_to(feature_set[interp_node_idx], type(interp_node[2]))
                            if not op(interp_feature_set_state, value):
                                return False, interp_probability
                            interp_probability.append(self.gene_interpolations[base_node]['probability'])
                        elif 'CHEMBL.COMPOUND' in self.feature_map[feature]:
                            base_node = (self.feature_map[feature], op, value)
                            interp_node = self.drug_interpolations[base_node]['interpolation']
                            interp_node_idx = self.features_list.index(interp_node[0])
                            interp_feature_set_state = try_convert_to(feature_set[interp_node_idx], type(interp_node[2]))
                            if not op(interp_feature_set_state, value):
                                return False, interp_probability
                            interp_probability.append(self.drug_interpolations[base_node]['probability'])
                        else:
                            return False, interp_probability
                else:
                    return False, interp_probability
            else:
                if feature_set[feature] != state_or_prop:
                    if (self.interp_type == 'gene' and self.gene_interpolations is not None) or (self.interp_type == 'drug' and self.drug_interpolations is not None):
                        #check bigram
                        if 'ENSEMBL' in self.feature_map[feature]:
                            base_node = (self.feature_map[feature], '==', feature_set[feature])
                            if base_node in self.gene_interpolations:
                                interp_node = self.gene_interpolations[base_node]['interpolation']
                                interp_node_idx = self.features_list.index(interp_node[0])
                                if feature_set[interp_node_idx] != interp_node[2]:
                                    return False, interp_probability
                            else:
                                return False, interp_probability
                            interp_probability.append(self.gene_interpolations[base_node]['probability'])
                        elif 'CHEMBL.COMPOUND' in self.feature_map[feature]:
                            base_node = (self.feature_map[feature], '==', feature_set[feature])
                            if base_node in self.drug_interpolations:
                                interp_node = self.drug_interpolations[base_node]['interpolation']
                                interp_node_idx = self.features_list.index(interp_node[0])
                                if feature_set[interp_node_idx] != interp_node[2]:
                                    return False, interp_probability
                            else:
                                return False, interp_probability
                            interp_probability.append(self.drug_interpolations[base_node]['probability'])
                        else:
                            return False, interp_probability
                    else:
                        return False, interp_probability
        return True, interp_probability

    def _is_consistent_with_target_state(self, target, state_or_prop, feature_set):
        if type(state_or_prop) == dict:
            op = get_operator(state_or_prop["op"])
            value = state_or_prop["value"]
            feature_set_state = try_convert_to(feature_set[target], type(value))
            if op(feature_set_state, value):
                return True
        else:
            if feature_set[target] == state_or_prop:
                return True
        return False

    def _discretize(self, data_dict, bins):
        continuous_feature_values = defaultdict(list)
        # Gather all continous variables
        for example, feature_dict in data_dict.items():
            for feature, state in feature_dict.items():
                if type(state) == int or type(state) == float:
                    #TODO: I am just going to convert to years here cause its easy this will need to be changed.
                    continuous_feature_values[feature].append(state / 365)
        # Get evenly spaced levels based on max and min values
        levels = {}
        for feature, values in continuous_feature_values.items():
            levels[feature] = list(np.round(np.linspace(min(values), max(values), num=bins-1), decimals=1))
        # Go back through patients and bin continuous values
        #print(levels)
        for example, feature_dict in data_dict.items():
            for feature in continuous_feature_values:
                #print(feature)
                #print(feature_dict[feature] / 365)
                _bin = np.digitize(feature_dict[feature] / 365, levels[feature])
                #print(_bin)
                if _bin == 0:
                    bin_name = '<{}'.format(levels[feature][0])
                elif _bin == bins-1:
                    bin_name = '>={}'.format(levels[feature][bins-1])
                else:
                    bin_name = '({}, {})'.format(levels[feature][_bin - 1], levels[feature][_bin])
                feature_dict[feature] = bin_name
                #print(bin_name)
                #input()
        return data_dict, levels
