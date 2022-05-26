import unittest
import os
import numpy as np
import pickle

from pybkb.bn import BN
from pybkb.exceptions import BKBNotMutexError


class BNApiTestCase(unittest.TestCase):
    def setUp(self):
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.wkdir, '../', 'data/sprinkler.dat'), 'rb') as data_file:
            self.data, self.feature_states, _ = pickle.load(data_file)

    def test_simple_build(self):
        bn1 = BN()
        rv_map = {
                'A': [1,2],
                1: ['B', 4],
                }
        # Add random variables
        for comp, states in rv_map.items():
            bn1.add_rv(comp, states)
        # Add Parents
        bn1.add_parents('A', [1])
        # Add some cpt entries
        bn1.add_cpt_entry('A', 1, .5, [(1, 'B')])
        bn1.add_cpt_entry('A', 2, .3, [(1, 'B')])
        bn1.add_cpt_entry('A', 1, .45, [(1, 4)])
        bn1.add_cpt_entry('A', 2, .7, [(1, 4)])
        bn1.add_cpt_entry(1, 'B', .2)
        bn1.add_cpt_entry(1, 4, .9)
        #bn1.save('test_bn_lib/test_api_bn1.bn')
        # Load saved bkb version
        bn2 = BN.load('test_bn_lib/test_api_bn1.bn')
        # Assert Equal
        self.assertEqual(bn1, bn2)

    def test_bkb_load(self):
        bkb = BN.load('test_bn_lib/test_api_bn1.bn')
    
    def test_eq(self):
        bn1 = BN.load('test_bn_lib/test_api_bn1.bn')
        bn2 = BN.load('test_bn_lib/test_api_bn1.bn')
        self.assertEqual(bn1, bn2)

    def test_generate_cpt_for_sprinkler(self):
        bn = BN()
        rv_map = {
                "cloudy": ['True', 'False'],
                "sprinkler": ['True', 'False'],
                "rain": ['True', 'False'],
                "wet_grass": ['True', 'False'],
                }
        # Add random variables
        for comp, states in rv_map.items():
            bn.add_rv(comp, states)
        # Add parents
        bn.add_parents('sprinkler', ['cloudy'])
        bn.add_parents('rain', ['cloudy'])
        bn.add_parents('wet_grass', ['sprinkler', 'cloudy'])
        # Use data to calculate CPT
        bn.calculate_cpts_from_data(self.data, self.feature_states)
    
    def test_make_bkb(self):
        bn = BN()
        rv_map = {
                "cloudy": ['True', 'False'],
                "sprinkler": ['True', 'False'],
                "rain": ['True', 'False'],
                "wet_grass": ['True', 'False'],
                }
        # Add random variables
        for comp, states in rv_map.items():
            bn.add_rv(comp, states)
        # Add parents
        bn.add_parents('sprinkler', ['cloudy'])
        bn.add_parents('rain', ['cloudy'])
        bn.add_parents('wet_grass', ['sprinkler', 'cloudy'])
        # Use data to calculate CPT
        bn.calculate_cpts_from_data(self.data, self.feature_states)
        d, m = bn.score(self.data, self.feature_states, 'mdl_ent', only='both')
        print(d, m, d+m)
        bn_bkb = bn.make_bkb()
        d, m = bn_bkb.score(self.data, self.feature_states, 'mdl_ent', only='both')
        print(d, m, d+m)

    def test_score_for_sprinkler(self):
        bn = BN()
        rv_map = {
                "cloudy": ['True', 'False'],
                "sprinkler": ['True', 'False'],
                "rain": ['True', 'False'],
                "wet_grass": ['True', 'False'],
                }
        # Add random variables
        for comp, states in rv_map.items():
            bn.add_rv(comp, states)
        # Add parents
        bn.add_parents('sprinkler', ['cloudy'])
        bn.add_parents('rain', ['cloudy'])
        bn.add_parents('wet_grass', ['sprinkler', 'cloudy'])
        # Use data to calculate CPT
        print(bn.score(self.data, self.feature_states, 'mdl_ent', only='both'))

    def test_from_bnlearn_modelstring(self):
        rv_map = {
                "cloudy": ['True', 'False'],
                "sprinkler": ['True', 'False'],
                "rain": ['True', 'False'],
                "wet_grass": ['True', 'False'],
                }
        bn = BN.from_bnlearn_modelstr(
                '[cloudy|rain][rain][sprinkler][wet_grass|sprinkler:rain]',
                rv_map,
                )
        print(bn.json())
