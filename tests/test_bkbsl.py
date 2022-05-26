import unittest
import pickle
import os
import random
random.seed(111)

from pybkb.learn import BKBLearner


class BKBSLTestCase(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.wkdir, '../', 'data/sprinkler.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = pickle.load(f_)

    def test_bkbsl_gobnilp_mdlent(self):
        learner = BKBLearner('gobnilp', 'mdl_ent', palim=1)
        learner.fit(self.data, self.feature_states, collapse=True)
        print(learner.report)
        #print(learner.report.bkf_data_scores)
        #print(learner.report.bkf_model_scores)

    def test_bkbsl_gobnilp_mdlmi(self):
        learner = BKBLearner('gobnilp', 'mdl_mi')
        learner.fit(self.data, self.feature_states)
        print(learner.report)
    
    def test_bkbsl_gobnilp_mdlent_distributed(self):
        learner = BKBLearner('gobnilp', 'mdl_ent', distributed=True, num_learners=10, num_cluster_nodes=1)
        learner.fit(self.data, self.feature_states)
        print(learner.report)
