import json
import time
import numpy as np


class LearningReport:
    def __init__(self, backend:str, rv_level:bool):
        """ Learning report class that keeps track of learning metrics like timing
            and scores based on backend and learning level.

        Args:
            :param backend: Name of learning backend: [gobnilp, notears].
            :type backend: str
            :param rv_level: Whether we are operating at the RV level (BN learning) or RV instantiation
                (BKB learning) level.
            :type rv_level: bool
        """
        self.backend = backend
        self.rv_level = rv_level
        self.start_time = time.time()
        # General metrics
        self.ncalls_to_joint = 0
        self.ncalls_to_jointhash = 0
        self.total_runtime = 0
        self.model_score = 0
        self.data_score = 0
        self.score = 0
        # BN metrics
        self.bn_learn_time = None
        self.like_bkb_data_score = None
        self.like_bkb_model_score = None
        self.like_bkb_score = None
        # BKB metrics
        self.bkf_learn_times = None
        self.bkf_ncalls_to_joint = None
        self.bkf_ncalls_to_jointhash = None
        self.bkf_model_scores = None
        self.bkf_data_scores = None
        self.bkf_scores = None
        self.bkf_modelstrings = None
        self.fusion_time = 0

    def to_dict(self):
        base = {
                "backend": self.backend,
                "ncalls to joint calculator": self.ncalls_to_joint,
                "ncalls saved due to hashing": self.ncalls_to_jointhash,
                "unique ncalls to joint calculator": self.unqiue_ncalls_to_joint,
                "total runtime": self.total_runtime,
                "scores": {
                    "data": self.data_score,
                    "model": self.model_score,
                    "total": self.score,
                    },
                }
        if self.rv_level:
            base["RV level"] = "RV (Bayesian Network)"
        else:
            base["RV level"] = "RV Instantiation (Bayesian Knowledge Base)"
        # Add in model specific reporting
        if self.bn_learn_time is not None:
            base["bn learn time"] = self.bn_learn_time
            base["like bkb scores"] = {
                    "data": self.like_bkb_data_score,
                    "model": self.like_bkb_model_score,
                    "total": self.like_bkb_score,
                    }
        else:
            base["average bkf learn time"] = self.average_bkf_learn_time
            base["average bkf model scores"] = self.average_bkf_model_scores
            base["average bkf data scores"] = self.average_bkf_data_scores
            base["average_bkf_scores"] = self.average_bkf_scores
            base["fusion time"] = self.fusion_time
            base["total bkb learn time"] = self.total_bkb_learn_time
        return base 

    def json(self, indent:int=2, filepath:str=None):
        d = self.to_dict()
        if filepath is not None:
            with open(filepath, 'w') as json_file:
                json.dump(d, json_file, indent=indent)
        return d

    def initialize_bkf_metrics(self, num_instances):
        self.bkf_learn_times = [0 for _ in range(num_instances)]
        self.bkf_ncalls_to_joint = [0 for _ in range(num_instances)]
        self.bkf_ncalls_to_jointhash = [0 for _ in range(num_instances)]
        self.bkf_model_scores = [0 for _ in range(num_instances)]
        self.bkf_data_scores = [0 for _ in range(num_instances)]
        self.bkf_scores = [0 for _ in range(num_instances)]
        self.bkf_modelstrings = ['' for _ in range(num_instances)]

    def update_from_store(self, store):
        self.ncalls_to_joint = store['__ncalls__']
        self.ncalls_to_jointhash = store['__nhashlookups__']

    def update_from_bkf_store(self, data_instance, cumulative_store):
        self.bkf_ncalls_to_joint[data_instance] = cumulative_store['__ncalls__'] - self.ncalls_to_joint
        self.bkf_ncalls_to_jointhash[data_instance] = cumulative_store['__nhashlookups__'] - self.ncalls_to_jointhash
        self.ncalls_to_joint = cumulative_store['__ncalls__']
        self.ncalls_to_jointhash = cumulative_store['__nhashlookups__']

    def add_bkf_metrics(self, data_instance, learn_time=None, model_score=None, data_score=None, bn=None):
        if learn_time is not None:
            self.bkf_learn_times[data_instance] = learn_time
        if model_score is not None:
            self.bkf_model_scores[data_instance] = model_score
        if data_score is not None:
            self.bkf_data_scores[data_instance] = data_score
        if data_score is not None and model_score is not None:
            self.bkf_scores[data_instance] = model_score + data_score
        if bn is not None:
            self.bkf_modelstrings[data_instance] = bn.bnlearn_modelstring()

    @property
    def average_bkf_learn_time(self):
        if self.bkf_learn_times is None:
            raise ValueError('No BKF learn times were added to the report')
        return np.average([t for t in self.bkf_learn_times])

    @property
    def average_bkf_model_scores(self):
        if self.bkf_model_scores is None:
            raise ValueError('No BKF model scores were added to the report')
        return np.average([s for s in self.bkf_model_scores])

    @property
    def average_bkf_data_scores(self):
        if self.bkf_data_scores is None:
            raise ValueError('No BKF data scores were added to the report')
        return np.average([s for s in self.bkf_data_scores])

    @property
    def average_bkf_scores(self):
        if self.bkf_scores is None:
            raise ValueError('No BKF scores were added to the report')
        return np.average([s for s in self.bkf_scores])

    @property
    def unqiue_ncalls_to_joint(self):
        return self.ncalls_to_joint - self.ncalls_to_jointhash

    def add_learned_bkb_scores(self, data_score, model_score):
        self.model_score = model_score
        self.data_score = data_score
        self.score = model_score + data_score

    def add_bn_like_bkb_scores(self, data_score, model_score):
        self.like_bkb_data_score = data_score
        self.like_bkb_model_score = model_score
        self.like_bkb_score = data_score + model_score

    def add_total_learn_time(self, learn_time):
        self.total_learn_time = learn_time
    
    def add_bn_learn_time(self, learn_time):
        self.bn_learn_time = learn_time

    @property
    def total_bkb_learn_time(self):
        return sum(self.bkf_learn_times)

    def str(self):
        report = "Learning Report\n"
        report += "-"*len(report) + "\n"
        report += f"Backend Used: {self.backend}\n"
        report += f"Learning Level: "
        if self.rv_level:
            report += "RV (Bayesian Network)\n"
        else:
            report += "RV Instantiation (BKB)\n"
        report += "Scores:\n"
        report += "++++++\n"
        report += f"Score = {self.score}\n"
        report += f"Model = {self.model_score}\n"
        report += f"Data = {self.data_score}\n"
        report += "Other Metrics:\n"
        report += "*************\n"
        report += f"Total Calls to Probability Calculator: {self.ncalls_to_joint}\n"
        report += f"Savings by storing Hash Table: {self.ncalls_to_jointhash}\n"
        return report

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def start_timer(self):
        self._start_learning = time.time()

    def end_timer(self):
        return time.time() - self._start_learning

    def finalize(self):
        self.total_runtime = time.time() - self.start_time
