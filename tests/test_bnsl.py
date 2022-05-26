import unittest
import pickle
import os
import random
random.seed(111)

from pybkb.learn import BNLearner


class BNSLTestCase(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.wkdir, '../', 'data/sprinkler.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = pickle.load(f_)

    def test_bnsl_gobnilp_mdlent(self):
        learner = BNLearner('gobnilp', 'mdl_ent')
        learner.fit(self.data, self.feature_states)
        #self.assertEqual(learner.bn.bnlearn_modelstring(), '[cloudy|rain][rain][sprinkler|cloudy][wet_grass|sprinkler:rain]')
        print(learner.report)

    def test_bnsl_gobnilp_mdlmi(self):
        learner = BNLearner('gobnilp', 'mdl_mi')
        learner.fit(self.data, self.feature_states)
        print(learner.report)
        #self.assertEqual(learner.bn.bnlearn_modelstring(), '[cloudy|wet_grass][wet_grass][rain|wet_grass][sprinkler|cloudy:rain]')
