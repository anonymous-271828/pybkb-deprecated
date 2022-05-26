import os
import sys
import math
import pandas as pd

from pybkb.common.bkb_grapher import make_graph

class Result:
    def __init__(self, result, bkb, evidence, marginal_evidence, targets, snode_list):
        self.raw_result = result
        self.bkb = bkb
        self.S_nodes = snode_list
        self.evidence = evidence
        self.marginal_evidence = evidence
        self.targets = targets


class RevisionResult(Result):
    def __init__(self, result, bkb, evidence=None, marginal_evidence=None, snode_list=None):
        super().__init__(result, bkb, evidence, marginal_evidence, None, snode_list)
        self.probabilities = result[0],
        self.contributions = result[1],
        self.completed_inferences = result[2]
        self.partials_explored = result[3]
        self.compute_time = result[4]

    def get_world_probablity(self):
        for _, prob in self.probabilities[0].items():
            return prob

    def summary(self, write_to_file=None):
        string = '-------- Revision Report --------\n'
        for res in self.result:
            string += 'Revision Answer {}\n'.format(res)
            string += '\tProbability = {}\n\tLog Prob = {}\n'.format(self.result[res]['Prob'],
                                                                     self.result[res]['Log Prob'])
            string += '\tWorld Instantiations:\n'
            for inode in self.result[res]['World']:
                if self.result[res]['World'][inode] is not None:
                    string += '\t\t{} = {}\n'.format(inode, self.result[res]['World'][inode])

        if write_to_file is not None:
            with open(write_to_file, 'w') as sum_file:
                sum_file.writeline(string)
        print(string)


class UpdatingResult(Result):
    def __init__(self, result, bkb, evidence=None, marginal_evidence=None, targets=None, snode_list=None):
        super(UpdatingResult, self).__init__(result, bkb, evidence, marginal_evidence, targets, snode_list)
        self.updates = result[0]
        self.contributions = result[1]
        self.completed_inferences = result[2]
        self.partials_explored = result[3]
        self.compute_time = result[4]
        self.meta_target_updates = None

    def number_of_inferences(self, target_name, target_state):
        target_comp_idx = self.bkb.getComponentIndex(target_name)
        target_state_idx = self.bkb.getComponentINodeIndex(target_comp_idx, target_state)

        inferences = self.completed_inferences[(target_comp_idx, target_state_idx)]

        if inferences is None:
            return 0
        return len(inferences)

    def get_inference(self, target_name, target_state, inference_idx):
        """ Extracts an inference from the completed_inferences results and returns
            the specifics in a human readable format.
        """
        target_comp_idx = self.bkb.getComponentIndex(target_name)
        target_state_idx = self.bkb.getComponentINodeIndex(target_comp_idx, target_state)

        return self.completed_inferences[(target_comp_idx, target_state_idx)][inference_idx]

    def graph_inference(
            self,
            inference,
            show=True,
            save_file=None,
            dpi=None,
            layout=None,
            size_multiplier=None,
            ):
        """ It takes an inference from a reasoning result and graphs it.
        """
        bkb_graph = make_graph(self.bkb)
        fig = bkb_graph.draw(
                inference=inference,
                layout=layout,
                show=show,
                save_file=save_file,
                dpi=dpi,
                size_multiplier=size_multiplier,
                )
        return bkb_graph, fig

    def process_completed_inferences(self):
        processed_inferences = dict()
        for (targ_comp_idx, targ_state_idx), evidence_inferences in self.completed_inferences.items():
            targ_comp_name = self.bkb.getComponentName(targ_comp_idx)
            targ_state_name = self.bkb.getComponentINodeName(targ_comp_idx, targ_state_idx)
            processed_inferences[(targ_comp_name, targ_state_name)] = dict()
            try:
                for evidence_set, inference_list in evidence_inferences.items():
                    for complete_inference_info in inference_list:
                        prob = complete_inference_info[0]
                        inodes = complete_inference_info[1]
                        processed_inodes = list()
                        for comp_idx, state_idx in inodes:
                            comp_name = self.bkb.getComponentName(comp_idx)
                            state_name = self.bkb.getComponentINodeName(comp_idx, state_idx)
                            processed_inodes.append((comp_name, state_name))
                        processed_inferences[(targ_comp_name, targ_state_name)][tuple(processed_inodes)] = prob
            except AttributeError:
                processed_inferences[(targ_comp_name,  targ_state_name)] = None
        return processed_inferences

    def completed_inferences_report(self):
        processed_inferences = self.process_completed_inferences()
        string = '-'*20 + ' Completed Inferences Report ' + '-'*20 + '\n'
        for (targ_comp, targ_state), inference_dict in processed_inferences.items():
            string += 'Target: {} = {}\n'.format(targ_comp, targ_state)
            if inference_dict is None:
                string += '\tNo inferences found for this target.\n'
                continue
            for inferences, prob in inference_dict.items():
                string += '\tWorld Probability = {}\n'.format(prob)
                string += '\tInstantiated I-nodes:\n'
                for inference in inferences:
                    string += '\t\t\t{} = {}\n'.format(inference[0], inference[1])
        return string

    def normalize_updates(self):
        normalized_updates = dict()
        total_prob = dict()
        for update, prob in self.updates.items():
            prob = max(0, prob)
            comp_idx, state_idx = update
            if comp_idx in total_prob:
                total_prob[comp_idx] += prob
            else:
                total_prob[comp_idx] = prob
        for update, prob in self.updates.items():
            if prob ==  -1:
                normalized_updates[update] = 0
                continue
            comp_idx, state_idx = update
            normalized_updates[update] = prob / total_prob[comp_idx]
        return normalized_updates

    def process_updates(self, normalize=False):
        if normalize:
            updates = self.normalize_updates()
        else:
            updates = self.updates
        processed_updates = dict()
        for update, prob in updates.items():
            comp_idx, state_idx = update
            comp_name = self.bkb.getComponentName(comp_idx)
            state_name = self.bkb.getComponentINodeName(comp_idx, state_idx)
            try:
                processed_updates[comp_name][state_name] = prob
            except:
                processed_updates[comp_name] = dict()
                processed_updates[comp_name][state_name] = prob
        if self.meta_target_updates is not None:
            processed_updates.update(self.meta_target_updates)
        return processed_updates

    def process_contributions(self):
        processed_contributions = dict()
        #print(self.contributions)
        #input()
        for inode_target, snodes_dict in self.contributions.items():
            if snodes_dict is not None:
                target_comp_idx, target_state_idx = inode_target
                target_comp_name = self.bkb.getComponentName(target_comp_idx)
                target_state_name = self.bkb.getComponentINodeName(target_comp_idx, target_state_idx)
                processed_contributions[(target_comp_name, target_state_name)] = dict()
                #print(target_comp_name, target_state_name)
                for snode_idx, contrib in snodes_dict.items():
                    snode = self.S_nodes[snode_idx]
                    #-- Get head
                    head_comp_idx, head_state_idx = snode.getHead()
                    head_comp_name = self.bkb.getComponentName(head_comp_idx)
                    head_state_name = self.bkb.getComponentINodeName(head_comp_idx, head_state_idx)
                    #print('Head:', head_comp_name, head_state_name)
                    if (head_comp_name, head_state_name) not in processed_contributions[(target_comp_name, target_state_name)]:
                        processed_contributions[(target_comp_name, target_state_name)][(head_comp_name, head_state_name)] = dict()
                    processed_tails = list()
                    for tail_idx in range(snode.getNumberTail()):
                        tail_comp_idx, tail_state_idx = snode.getTail(tail_idx)
                        tail_comp_name = self.bkb.getComponentName(tail_comp_idx)
                        tail_state_name = self.bkb.getComponentINodeName(tail_comp_idx, tail_state_idx)
                        processed_tails.append((tail_comp_name, tail_state_name))
                    processed_contributions[(target_comp_name, target_state_name)][(head_comp_name, head_state_name)][tuple(processed_tails)] = contrib
                    #print('Tail:', processed_tails)
                    #print(processed_contributions)
                    #input('Next')
        return processed_contributions

    def print_contributions(self):
        processed_contributions = self.process_contributions()
        string = '-'*20 + ' S-nodes Contribution Report ' + '-'*20 + '\n'
        for (targ_comp_name, targ_state_name), head_tail_contrib in processed_contributions.items():
            string += 'Target: {} = {}\n'.format(targ_comp_name, targ_state_name)
            for (head_comp, head_state), tail_contrib in head_tail_contrib.items():
                string += '\tHead: {} = {}\n'.format(head_comp, head_state)
                for tails, contrib in tail_contrib.items():
                    string += '\tTail:\n'
                    for tail_comp, tail_state in tails:
                        string += '\t\t{} = {}\n'.format(tail_comp, tail_state)
                    string += '\tContribution = {}\n'.format(contrib)
        return string

    @staticmethod
    def isSourceComponent(comp_name):
        if 'Source' in comp_name or 'Collection' in comp_name:
            return True
        return False

    def process_inode_contributions(self, include_srcs=True, top_n_inodes=None, ignore_prefixes=None, remove_tuples=False):
        """ Processes I-node contributions to the inference.

        Parameters
        ----------
        include_srcs: bool, optional
            Filters out Source nodes from returned contributions.

        top_n_inodes: int, optional
            Returns only the top n contrinuting I-nodes for each given inference target.

        ignore_prefixes: list, optional
            A list of string prefixes that will be ignorned when processing contributions.

        remove_tuples: bool, optional
            Will hash on a str verison of a tuple if a tuple is found.

        Returns
        -------

        processed_inode_contrib: dict
            A nested dictionary of form: [target][inode] = contribution
        """
        processed_inode_contrib = dict()
        for target, snode_dict in self.process_contributions().items():
            if remove_tuples:
                target = '{} = {}'.format(target[0], target[1])
            processed_inode_contrib[target] = dict()
            for head, tail_dict in snode_dict.items():
                head_comp_name, head_state_name = head
                if remove_tuples:
                    head = '{} = {}'.format(head_comp_name, head_state_name)
                if not include_srcs:
                    if self.isSourceComponent(head_comp_name):
                        continue
                if ignore_prefixes is not None:
                    ignore = False
                    for prefix in ignore_prefixes:
                        if head_comp_name[:len(prefix)] == prefix and not self.isSourceComponent(head_comp_name):
                            ignore = True
                            break
                    if ignore:
                        continue
                if head in processed_inode_contrib[target]:
                    #-- Add contribution
                    for tail, contrib in tail_dict.items():
                        processed_inode_contrib[target][head] += contrib
                else:
                    #-- Add contributions
                    processed_inode_contrib[target][head] = 0
                    for tail, contrib in tail_dict.items():
                        processed_inode_contrib[target][head] += contrib
            if top_n_inodes is not None:
                #-- Only return top n inodes with the highest contribution
                #-- Rank and sort.
                sort_contribs = sorted([(contrib, inode) for inode, contrib in processed_inode_contrib[target].items()], reverse=True)
                filter_inode_contribs = {inode: contrib for contrib, inode in sort_contribs[:top_n_inodes]}
                #-- Replaced with filter dictionary.
                processed_inode_contrib[target] = filter_inode_contribs
        return processed_inode_contrib

    def contribs_to_dataframes(self, inode_dict, scale=False):
        dfs = dict()
        for target, contrib_dict in inode_dict.items():
            data_dict = {'I-node': list(), 'Contribution': list()}
            for inode, contrib in contrib_dict.items():
                data_dict['I-node'].append('{} = {}'.format(*inode))
                data_dict['Contribution'].append(contrib)
            df = pd.DataFrame(data=data_dict)
            df.sort_values(by=['Contribution'], inplace=True, ascending=False)
            df.set_index(['I-node'], inplace=True)
            if scale:
                df['Contribution'] = 100 * ((df['Contribution'] - df['Contribution'].mean()) / df['Contributions'].std())
            dfs[target] = df
        return dfs

    def summary(self, write_to_file=None, include_srcs=True, normalize=False, include_contributions=True):
        string = '-------- Updating Report --------\n'
        for comp_name, state_prob_dict in self.process_updates(normalize).items():
            string += '{}\n'.format(comp_name)
            for state_name, prob in state_prob_dict.items():
                string += '\t{} = {}\n'.format(state_name, prob)

        if include_contributions:
            string += '\nContribution Analysis:\n'
            inode_contribs = self.process_inode_contributions(include_srcs)
            contrib_dfs = self.contribs_to_dataframes(inode_contribs)
            for (comp_name, state_name), df in contrib_dfs.items():
                string += 'Target I-Node: {} = {}\n'.format(comp_name, state_name)
                string += df.to_string()
                string += '\n' + '-'*30 + '\n'

        if write_to_file is not None:
            with open(write_to_file, 'w') as sum_file:
                sum_file.writeline(string)
        print(string)
