import unittest
import os

from pybkb.bkb import BKB
from pybkb.fusion import fuse

class FusionTestCase(unittest.TestCase):
    def setUp(self):
        self.wkdir = os.path.dirname(os.path.abspath(__file__))

    def test_pirate_fusion(self):
        bkb_paths = [
                'pybkb/bkbs/fisherman.bkb',
                'pybkb/bkbs/illegal_dumping_ev.bkb',
                'pybkb/bkbs/illegal_fishing_ev.bkb',
                'pybkb/bkbs/pirate.bkb',
                'pybkb/bkbs/tsunami_ev.bkb',
                ]
        # Load bkbs
        bkfs = []
        for path in bkb_paths:
            bkfs.append(BKB.load(os.path.join(self.wkdir, '../', path)))
        # Fuse
        bkb = fuse(bkfs, [1 for _ in range(len(bkfs))], collapse=False)
        #print(bkb.json())
        self.assertTrue(bkb.is_mutex())

    def test_goldfish_fusion(self):
        bkb_paths = [
                'pybkb/bkbs/goldfish.bkb',
                ]*100
        # Load bkbs, need to add different source names as these are all copies
        bkfs = []
        srcs = []
        for idx, path in enumerate(bkb_paths):
            bkfs.append(BKB.load(os.path.join(self.wkdir, '../', path)))
            srcs.append(str(idx))
        # Fuse
        bkb = fuse(bkfs, [1 for _ in range(len(bkfs))], source_names=srcs, collapse=False, verbose=True)
        #self.assertTrue(bkb.is_mutex(verbose=True))
    
    def test_goldfish_fusion_collapse(self):
        bkb_paths = [
                'pybkb/bkbs/goldfish.bkb',
                ]*100
        # Load bkbs, need to add different source names as these are all copies
        bkfs = []
        srcs = []
        for idx, path in enumerate(bkb_paths):
            bkfs.append(BKB.load(os.path.join(self.wkdir, '../', path)))
            srcs.append(str(idx))
        # Fuse
        bkb = fuse(bkfs, [1 for _ in range(len(bkfs))], source_names=srcs, collapse=True, verbose=True)
        #self.assertTrue(bkb.is_mutex(verbose=True))
