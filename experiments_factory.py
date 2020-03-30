import json
import os
import shutil
from typing import Dict, List

import datasets_loading_factory
import derivatives_factory
import metrics_factory
import models_factory
import optimisation_factory
import rnn_training_experiments

EXPERIMENT_FACTORY_PATH = './experiments_factory.json'

DATASET_HYPERPARAMETERS = 'dataset_hyperparameters'
EXPERIMENT_NAME = 'experiment_name'
K_HYPERPARAMETERS = 'k_matrix_hyperparameters'
H_HYPERPARAMETERS = 'hessian_hyperparameters'
METRICS_HYPERPARAMETERS = 'metrics_hyperparameters'
MODEL_HYPERPARAMETERS = 'model_hyperparameters'
OPTIMISER_HYPERPARAMETERS = 'optimiser_hyperparameters'
TRAINING_HYPERPARAMETERS = 'training_hyperparameters'


def make_dir(name: str):
    if not os.path.exists(name):
        os.makedirs(name)


def get_experiments_outline() -> List[Dict]:
    print('Reading experiments outline from: {}'.format(EXPERIMENT_FACTORY_PATH))
    with open(EXPERIMENT_FACTORY_PATH, 'r') as f:
        experiments_outline = json.load(f)
    return experiments_outline


def get_new_model_path(experiment_name: str) -> str:
    return '{}/{}_MODEL.pt'.format(experiment_name, experiment_name)


def copy_file(old_path: str, new_path: str) -> None:
    shutil.copyfile(old_path, new_path)


def save_experiment_dict(experiment_dict_):
    dir_files_names = os.listdir(experiment_dict_[EXPERIMENT_NAME])
    dir_files_names_filtered = list(filter(lambda x: 'arguments' in x and 'json' in x, dir_files_names))
    dumped_dict = json.dumps(experiment_dict_)
    file_name = '{}/arguments_{}.json'.format(experiment_dict_[EXPERIMENT_NAME], len(dir_files_names_filtered))
    f = open(file_name, "w")
    f.write(dumped_dict)
    f.close()


def init_experiment_env(experiment_dict_):
    name = experiment_dict_[EXPERIMENT_NAME]
    print('Initialising an environment for the experiment: {}'.format(name))
    make_dir(name=name)
    save_experiment_dict(experiment_dict_=experiment_dict_)


def perform_experiment(experiment_dict_):
    train_model(experiment_dict_=experiment_dict_)


def train_model(experiment_dict_: Dict):
    dataset_generator = datasets_loading_factory.DatasetGeneratorPyTorch(
        **experiment_dict_[DATASET_HYPERPARAMETERS],
    )
    k_matrix_factory = derivatives_factory.KMatrixFactory(
        dataset_generator=dataset_generator, **experiment_dict[K_HYPERPARAMETERS]
    )
    hessian_factory = derivatives_factory.HessianFactory(
        dataset_generator=dataset_generator, **experiment_dict[H_HYPERPARAMETERS]
    )
    model_factory_ = models_factory.ModelsFactory(**experiment_dict_[MODEL_HYPERPARAMETERS])
    optimiser_factory = optimisation_factory.OptimisersFactoryPyTorch(**experiment_dict_[OPTIMISER_HYPERPARAMETERS])
    metrics_factory_ = metrics_factory.MetricsFactory(**experiment_dict_[METRICS_HYPERPARAMETERS])
    training_experiment = rnn_training_experiments.RNNTrainingExperimentPyTorch(
        **experiment_dict_[TRAINING_HYPERPARAMETERS],
        dataset_generator=dataset_generator,
        models_factory=model_factory_,
        optimisers_factory=optimiser_factory,
        metrics_factory=metrics_factory_,
        k_matrix_factory=k_matrix_factory,
        hessian_factory=hessian_factory,
    )
    training_experiment.run()


if __name__ == '__main__':
    experiments_outline = get_experiments_outline()
    print('Starting performing experiments.')
    for experiment_dict in experiments_outline:
        init_experiment_env(experiment_dict_=experiment_dict)
        perform_experiment(experiment_dict_=experiment_dict)
