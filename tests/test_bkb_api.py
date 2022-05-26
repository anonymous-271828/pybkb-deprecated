import unittest
import os
import numpy as np

from pybkb.bkb import BKB
from pybkb.exceptions import BKBNotMutexError


class BKBApiTestCase(unittest.TestCase):
    def setUp(self):
        self.wkdir = os.path.dirname(os.path.abspath(__file__))

    def test_simple_build(self):
        bkb1 = BKB()
        inodes = [('A', 1), ('A', '2'), (1, 1), (1, 'B')]
        for comp, state in inodes:
            bkb1.add_inode(comp, state)
        # Add S-nodes
        bkb1.add_snode('A', 1, .003)
        bkb1.add_snode(1,1,.9,[('A',1)])
        bkb1.add_snode(1,'B',.56)
        bkb1.add_snode('A', '2', .97, [(1,1)])
        #bkb1.save('test_bkb_lib/test_api_bkb1.bkb')
        # Load saved bkb version
        bkb2 = BKB.load('test_bkb_lib/test_api_bkb1.bkb')
        # Assert Equal
        self.assertEqual(bkb1, bkb2)

    def test_bkb_load(self):
        bkb = BKB.load('test_bkb_lib/test_api_bkb1.bkb')
    
    def test_bkb_load_legacy(self):
        bkb1 = BKB.load_legacy(
            os.path.join(self.wkdir, '../', 'examples/aquatic_eco.bkb'),
            use_pickle=False,
            compress=False,
            )
        bkb2 = BKB.load_legacy(
            os.path.join(self.wkdir, '../', 'examples/aquatic_eco_binary.bkb'),
            use_pickle=True,
            compress=True,
            )
        self.assertEqual(bkb1, bkb2)
        #bkb1.save(os.path.join(self.wkdir, '../', 'pybkb/bkbs/aquatic_eco.bkb'))

    def test_eq(self):
        bkb1 = BKB.load('test_bkb_lib/test_api_bkb1.bkb')
        bkb2 = BKB.load('test_bkb_lib/test_api_bkb1.bkb')
        self.assertEqual(bkb1, bkb2)

    def test_is_mutex(self):
        bkb = BKB.load('test_bkb_lib/test_api_bkb1.bkb')
        self.assertTrue(bkb.is_mutex())
        # Make a non mutex bkb
        bkb = BKB()
        inodes = [(1, True), (1, False), (2, True), (2, False)]
        for comp, state in inodes:
            bkb.add_inode(comp, state)
        bkb.add_snode(1, True, .3)
        bkb.add_snode(1, True, .4)
        self.assertRaises(BKBNotMutexError, bkb.is_mutex)
        # Make another non mutex bkb
        bkb = BKB()
        inodes = [(1, True), (1, False), (2, True), (2, False)]
        for comp, state in inodes:
            bkb.add_inode(comp, state)
        bkb.add_snode(1, True, .3, [(2, True)])
        bkb.add_snode(1, True, .4)
        self.assertRaises(BKBNotMutexError, bkb.is_mutex)
        # Make another non mutex bkb
        bkb = BKB()
        inodes = [(1, True), (1, False), (2, True), (2, False), (3, True), (3, False)]
        for comp, state in inodes:
            bkb.add_inode(comp, state)
        bkb.add_snode(1, True, .3, [(2, True)])
        bkb.add_snode(1, True, .4, [(3, False)])
        self.assertRaises(BKBNotMutexError, bkb.is_mutex)

    def test_crs(self):
        bkb = BKB.load('test_bkb_lib/test_api_bkb1.bkb')
        crs = bkb.get_causal_ruleset()
        # Make sure there are only 2 feature keys as expected
        self.assertEqual(len(crs), 2)
        # Makre sure that the s-node indices is equal to the total number of s-nodes in the bkb
        total_crs_snodes = sum([len(values) for _, values in crs.items()])
        self.assertEqual(total_crs_snodes, len(bkb.snode_probs))

    def test_find_snodes(self):
        bkb = BKB.load(
            os.path.join(self.wkdir, '../', 'pybkb/bkbs/aquatic_eco.bkb'),
            )
        self.assertEqual(len(bkb.find_snodes('ReproduceAbility', 'Low')), 2)
        self.assertEqual(len(bkb.find_snodes('ReproduceAbility', 'Low', tail_subset=[("pH_Value","Normal")])), 1)
        self.assertEqual(len(bkb.find_snodes('ReproduceAbility', 'Low', prob=0.1, tail_subset=[("pH_Value","Normal")])), 1)
