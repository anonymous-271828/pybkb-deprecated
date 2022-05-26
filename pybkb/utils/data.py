import os
import csv
import compress_pickle
import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
import numpy as np
from collections import defaultdict


class DataWrangler:
    def __init__(
            self,
            feature_names:list=None,
            train_data_path:str=None,
            test_data_path:str=None,
            predict_class_name:str=None,
            has_feature_header:bool=True,
            source_labels_feature:str=None,
            train_data:list=None,
            test_data:list=None,
            ):
        """ A generic data wrangler class to assist in building data objects for 
        structure learning.

        Kwargs:
            :param feature_names: Name of features in the dataset, should exactly the same as the
                associated column in the data file.
            :type feature_names: list
            :param train_data_path: Path to the training data, if any.
            :type train_data_path: str
            :param test_data_path: Path to the test data, if any.
            :type test_data_path: str
            :param predict_class_name: Name of prediction feature.
            :type predict_class_name: str
            :param has_feature_header: Whether or not data has a header associated to the feature columns in the dataset.
            :type has_feature_header: bool
        """
        self.features = feature_names
        self.train_path = train_data_path
        self.test_path = test_data_path
        if train_data is None and test_data is None:
            self.train_data, self.test_data, features, self.train_srcs, self.test_srcs = self.load_data(
                    self.train_path,
                    self.test_path,
                    has_feature_header,
                    source_labels_feature,
                    feature_names,
                    )
        else:
            self.train_data, self.test_data, self.train_srcs, self.test_srcs = self.initialize_data(
                    train_data,
                    test_data,
                    )
        if feature_names is not None:
            self.features = feature_names
        else:
            self.features = features
        self.data = self.train_data + self.test_data
        self.predict_class = predict_class_name
        self.table = self.convert_to_orange(self.data)

    @staticmethod
    def initialize_data(
            train_data:list=None,
            test_data:list=None,
            ):
        train_srcs = []
        test_srcs = []
        if train_data is not None:
            train_srcs = [f'train-{i}' for i in range(len(train_data))]
        if test_data is not None:
            test_srcs = [f'test-{i}' for i in range(len(test_data))]
        return train_data, test_data, train_srcs, test_srcs

    @staticmethod
    def load_data(
            train_path:str,
            test_path:str,
            has_feature_header:bool,
            source_labels_feature:str=None,
            features:list=None
            ):
        train = []
        train_srcs = []
        test = []
        test_srcs = []
        train_features = features
        test_features = features
        if train_path is None and test_path is None:
            raise ValueError('Need to specify either train or test data path.')
        if train_path:
            with open(train_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                for idx, row in enumerate(reader):
                    if has_feature_header and idx == 0:
                        train_features = row
                        continue
                    row = [x.strip() for x in row]
                    if source_labels_feature is not None:
                        source_label_idx = train_features.index(source_labels_feature)
                        train_srcs.append(row[source_label_idx])
                    else:
                        source_label_idx = -1
                        train_srcs.append(f'train-{idx}')
                    train.append([x for i, x in enumerate(row) if i != source_label_idx])
        if test_path:
            with open(test_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                for idx, row in enumerate(reader):
                    if has_feature_header and idx == 0:
                        test_features = row
                        continue
                    row = [x.strip() for x in row]
                    if source_labels_feature is not None:
                        source_label_idx = test_features.index(source_labels_feature)
                        test_srcs.append(row[source_label_idx])
                    else:
                        source_label_idx = -1
                        train_srcs.append(f'test-{idx}')
                    test.append([x for i, x in enumerate(row) if i != source_label_idx])
        # Clean the data
        if len(train) > 0:
            train = DataWrangler.clean(train)
        if len(test) > 0:
            test = DataWrangler.clean(test)
        # Compare features
        features = None
        if has_feature_header:
            if train_features and test_features and train_features != test_features:
                raise ValueError('Header on train and test dataset do not match.')
            elif train_features:
                features = train_features
            elif test_features:
                features = test_features
        if source_labels_feature:
            features.remove(source_labels_feature)
        return train, test, features, train_srcs, test_srcs

    @staticmethod 
    def clean(data:list):
        # Remove any empty rows
        data_ls_cleaner = []
        for row in data:
            if len(row) != 0:
                data_ls_cleaner.append(row)
        # Check to make sure all rows are the same length
        row_lens = []
        for row in data_ls_cleaner:
            row_lens.append(len(row))
        assert len(set(row_lens)) == 1
        return data_ls_cleaner
    
    def convert_to_orange(self, data:list):
        # Get feature states dict and types
        feature_states, variable_types = self.collect_feature_states_and_var_types(data)
        feature_vars = []
        for feature, states in feature_states.items():
            # Skip if prediction class
            if self.predict_class is not None and feature == self.predict_class:
                if variable_types[feature] == 'disc':
                    class_var = DiscreteVariable.make(feature, values=list(states))
                else:
                    class_var = ContinuousVariable.make(feature)
                continue
            if variable_types[feature] == 'disc':
                feature_vars.append(DiscreteVariable.make(feature, values=list(states)))
            else:
                feature_vars.append(ContinuousVariable.make(feature))
        # Create an all numerical data array using the indices in the values of the discrete variables
        data_numerical = []
        for row in data:
            new_row = []
            for feature_var, state in zip(feature_vars + [class_var], row):
                if feature_var.is_continuous:
                    new_row.append(float(state))
                else:
                    new_row.append(feature_var.values.index(state))
            data_numerical.append(new_row)
        # Convert to numpy array
        data_np = np.asarray(data_numerical)
        # Setup domain
        domain = Domain(feature_vars, class_var)
        # Convert this to a orange table
        table = Orange.data.Table.from_numpy(domain, X=data_np[:,:-1], Y=data_np[:,-1])
        return table

    def collect_feature_states_and_var_types(self, data:list):
        feature_states = defaultdict(set)
        variable_types = {}
        for row in data:
            for feature, state in zip(self.features, row):
                try:
                    state = float(state)
                    if feature in variable_types:
                        assert variable_types[feature] == 'cont'
                    else:
                        variable_types[feature] = 'cont'
                except:
                    if feature in variable_types:
                        assert variable_types[feature] == 'disc'
                    else:
                        variable_types[feature] = 'disc'
                feature_states[feature].add(state)
        return feature_states, variable_types

    def discretize(self, disc_method:str='entorpy_mdl'):
        # Setup discretization
        disc = Orange.preprocess.Discretize()
        if disc_method == 'entropy_mdl':
            disc.method = Orange.preprocess.discretize.EntropyMDL()
        else:
            raise ValueError('Unknown discretization method.')
        # Discretize
        return disc(self.table)

    def get_bkb_dataset(self, discretize:bool=True, disc_method:str='entropy_mdl', combine_train_test:bool=False):
        if discretize:
            table = self.discretize(disc_method)
        else:
            table = self.table
        # Construct new feature states from table domain plus maps for speed
        feature_states = []
        fs_map = {}
        fs_map_repr = {}
        i = 0
        for feature_var in list(table.domain.attributes) + list(table.domain.class_vars):
            for value in feature_var.values:
                feature_state = (feature_var.name, value)
                feature_states.append(feature_state)
                fs_map[feature_state] = i
                fs_map_repr[i] = feature_state
                i += 1
        # Construct BKB learning style dataset one-hot encoded over feature_states
        bkb_train_data = []
        bkb_test_data = []
        is_train_data = True
        for row_idx, row in enumerate(table):
            if row_idx > len(self.train_data) - 1:
                is_train_data = False
            new_row = np.zeros(len(feature_states))
            for feature_var, value in zip(table.domain.attributes, row.x):
                new_row[fs_map[(feature_var.name, feature_var.repr_val(value))]] = 1
            for class_var, value in zip(table.domain.class_vars, row.y):
                new_row[fs_map[(class_var.name, class_var.repr_val(value))]] = 1
            if is_train_data:
                bkb_train_data.append(new_row)
            else:
                bkb_test_data.append(new_row)
        if combine_train_test:
            return np.asarray(bkb_train_data + bkb_test_data), feature_states, self.train_srcs + self.test_srcs
        return np.asarray(bkb_train_data), np.asarray(bkb_test_data), feature_states, self.train_srcs, self.test_srcs


class KeelWrangler(DataWrangler):
    def __init__(self, data_path:str, compression:str=None):
        self.data_path = data_path
        self.compression = compression
        train_data, test_data, self.header = self.load_keel_dataset()
        feature_names = [feature for (feature, values) in self.header["attributes"]]
        predict_class = self.header["outputs"][0]
        super().__init__(
                feature_names=feature_names,
                predict_class_name=predict_class,
                train_data=train_data,
                test_data=test_data,
                )

    def load_keel_dataset(self):
        with open(self.data_path, 'rb') as keel_file:
            if self.compression is not None:
                data, header = compress_pickle.load(keel_file, compression=self.compression)
            else:
                data, header = pickle.load(keel_file)
        # Extract train and test (no cross validation so just take the first list item)
        X_train, Y_train, X_test, Y_test = data[0]
        # Merge X and Ys
        if X_train is not None:
            train_data = np.concatenate((X_train, Y_train), axis=1).tolist()
        else:
            train_data = []
        if X_test is not None:
            test_data = np.concatenate((X_test, Y_test), axis=1).tolist()
        else:
            test_data = []
        return train_data, test_data, header

if __name__ == '__main__':
    path = '/mnt/c/Users/c_yak/Downloads/music_features_bench/data.csv'
    wrangler = DataWrangler(train_data_path=path, predict_class_name='label', source_labels_feature='filename')
    print(wrangler.features)
    data, feature_states, srcs = wrangler.get_bkb_dataset(combine_train_test=True)
    print(data)
    print(feature_states)
    print(srcs)
