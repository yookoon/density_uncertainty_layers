import os
import requests
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch


UCI_DATASETS = [
    "BostonHousing", 
    "ConcreteStrength", 
    "EnergyEfficiency",
    "Kin8nm",
    "NavalPropulsion",
    "ProteinStructure",
    "WineQuality",
    "YachtHydrodynamics",
    "YearPredictionMSD",
    ]


def download(cache_dir, name, url):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_path = os.path.join(cache_dir, name)
    if not os.path.exists(data_path):
        print(f"Download dataset to {cache_dir}")
        response = requests.get(url)
        with open(data_path, "wb") as f:
            f.write(response.content)

    return data_path

def normalize(train, test):
    # Normalize using the training set statistics
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    
    train = np.nan_to_num((train - mean) / std)
    test = np.nan_to_num((test - mean) / std)
    return train, test


class UCIDataset(torch.utils.data.Dataset):
    n_split = 20
    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)


class BostonHousing(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    
    def __init__(self, root, split_id, train):
        assert split_id < self.n_split
        cache_dir = os.path.join(root, "boston_housing")
        data_path = download(cache_dir, "housing.data", self.url)
        table = pd.read_table(data_path, names=None, header=None, delim_whitespace=True).to_numpy()

        trainset, testset = train_test_split(table, test_size=0.1, random_state=split_id)
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        train_data, train_targets = trainset[:, :-1], trainset[:, -1]
        test_data, test_targets = testset[:, :-1], testset[:, -1]

        self.target_std = train_targets.std()
        train_data, test_data = normalize(train_data, test_data)
        train_targets, test_targets = normalize(train_targets, test_targets)

        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets


class ConcreteStrength(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

    def __init__(self, root, split_id, train):
        assert split_id < self.n_split
        cache_dir = os.path.join(root, "concrete_strength")
        data_path = download(cache_dir, "concrete.xls", self.url)
        table = pd.read_excel(data_path).to_numpy()

        trainset, testset = train_test_split(table, test_size=0.1, random_state=split_id)
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        train_data, train_targets = trainset[:, :-1], trainset[:, -1]
        test_data, test_targets = testset[:, :-1], testset[:, -1]

        self.target_std = train_targets.std()
        train_data, test_data = normalize(train_data, test_data)
        train_targets, test_targets = normalize(train_targets, test_targets)
        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets


class EnergyEfficiency(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"

    def __init__(self, root, split_id, train):
        assert split_id < self.n_split
        cache_dir = os.path.join(root, "energy_efficiency")
        data_path = download(cache_dir, "energy_efficiency.xls", self.url)
        table = pd.read_excel(data_path).to_numpy()

        trainset, testset = train_test_split(table, test_size=0.1, random_state=split_id)
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        train_data, train_targets = trainset[:, :-2], trainset[:, -1]
        test_data, test_targets = testset[:, :-2], testset[:, -1]

        self.target_std = train_targets.std()
        train_data, test_data = normalize(train_data, test_data)
        train_targets, test_targets = normalize(train_targets, test_targets)
        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets


class Kin8nm(UCIDataset):
    url = "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff"

    def __init__(self, root, split_id, train):
        assert split_id < self.n_split
        cache_dir = os.path.join(root, "kin8nm")
        data_path = download(cache_dir, "kin8nm.csv", self.url)
        table = pd.read_csv(data_path).to_numpy()

        trainset, testset = train_test_split(table, test_size=0.1, random_state=split_id)
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        train_data, train_targets = trainset[:, :-1], trainset[:, -1]
        test_data, test_targets = testset[:, :-1], testset[:, -1]

        self.target_std = train_targets.std()
        train_data, test_data = normalize(train_data, test_data)
        train_targets, test_targets = normalize(train_targets, test_targets)
        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets


class NavalPropulsion(UCIDataset):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip"

    def __init__(self, root, split_id, train):
        assert split_id < self.n_split
        cache_dir = os.path.join(root, "naval_propulsion")
        data_path = download(cache_dir, "naval_propulsion.zip", self.url)
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
        data_path = os.path.join(cache_dir, "UCI CBM Dataset", "data.txt")
        table = pd.read_csv(data_path, delim_whitespace=True, header=None).to_numpy()

        trainset, testset = train_test_split(table, test_size=0.1, random_state=split_id)
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        train_data, train_targets = trainset[:, :-2], trainset[:, -2]
        test_data, test_targets = testset[:, :-2], testset[:, -2]

        self.target_std = train_targets.std()
        train_data, test_data = normalize(train_data, test_data)
        train_targets, test_targets = normalize(train_targets, test_targets)
        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets


class ProteinStructure(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
    n_split = 5

    def __init__(self, root, split_id, train):
        assert split_id < self.n_split
        cache_dir = os.path.join(root, "protein_structure")
        data_path = download(cache_dir, "CASP.csv", self.url)
        table = pd.read_csv(data_path).to_numpy()
        trainset, testset = train_test_split(table, test_size=0.1, random_state=split_id)
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        train_data, train_targets = trainset[:, 1:], trainset[:, 0]
        test_data, test_targets = testset[:, 1:], testset[:, 0]

        self.target_std = train_targets.std()
        train_data, test_data = normalize(train_data, test_data)
        # Do not normalize the test targets as it will affect the RMSE and LL 
        train_targets, test_targets = normalize(train_targets, test_targets)
        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets


class WineQuality(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    def __init__(self, root, split_id, train):
        assert split_id < self.n_split
        cache_dir = os.path.join(root, "wine_quality")
        data_path = download(cache_dir, "winequality-red.csv", self.url)
        table = pd.read_csv(data_path, sep=";").to_numpy()
        
        trainset, testset = train_test_split(table, test_size=0.1, random_state=split_id)
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        train_data, train_targets = trainset[:, :-1], trainset[:, -1]
        test_data, test_targets = testset[:, :-1], testset[:, -1]

        self.target_std = train_targets.std()
        train_data, test_data = normalize(train_data, test_data)
        train_targets, test_targets = normalize(train_targets, test_targets)
        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets


class YachtHydrodynamics(UCIDataset):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"

    def __init__(self, root, split_id, train):
        assert split_id < self.n_split
        cache_dir = os.path.join(root, "yacht_hydrodynamics")
        data_path = download(cache_dir, "yacht_hydrodynamics.data", self.url)
        table = pd.read_csv(data_path, delim_whitespace=True, header=None).to_numpy()

        trainset, testset = train_test_split(table, test_size=0.1, random_state=split_id)
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        train_data, train_targets = trainset[:, :-1], trainset[:, -1]
        test_data, test_targets = testset[:, :-1], testset[:, -1]

        self.target_mean = train_targets.mean()
        self.target_std = train_targets.std()
        train_data, test_data = normalize(train_data, test_data)
        train_targets, test_targets = normalize(train_targets, test_targets)
        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets


class YearPredictionMSD(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
    n_split = 1

    def __init__(self, root, split_id, train):
        assert split_id < self.n_split
        cache_dir = os.path.join(root, "year_prediction_msd")
        data_path = download(cache_dir, "YearPredictionMSD.txt.zip", self.url)
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
        data_path = os.path.join(cache_dir, "YearPredictionMSD.txt")
        table = pd.read_csv(data_path, header=None).to_numpy()

        trainset, testset = train_test_split(table, test_size=0.1, random_state=split_id)
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        train_data, train_targets = trainset[:, 1:], trainset[:, 0]
        test_data, test_targets = testset[:, 1:], testset[:, 0]

        self.target_std = train_targets.std()
        train_data, test_data = normalize(train_data, test_data)
        train_targets, test_targets = normalize(train_targets, test_targets)
        if train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets



if __name__ == "__main__":
    for dataset_name in UCI_DATASETS:
        dataset = globals()[dataset_name]('./data', 0, train=True)
        print(f'{dataset_name} n_split:{dataset.n_split}, Data shape: {dataset.data.shape}')