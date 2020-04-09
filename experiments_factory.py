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

DATA_PATH = 'data'


def make_dir(name: str):
    if not os.path.exists(name):
        os.makedirs(name)


def get_experiments_outline() -> List[Dict]:
    print('Reading experiments outline from: {}'.format(EXPERIMENT_FACTORY_PATH))
    with open(EXPERIMENT_FACTORY_PATH, 'r') as f:
        experiments_outline = json.load(f)
    return experiments_outline


def save_experiment_dict(experiment_dict_, path):
    dir_files_names_filtered = list(filter(lambda x: 'arguments' in x and 'json' in x, path))
    dumped_dict = json.dumps(experiment_dict_)
    file_name = '{}/arguments_{}.json'.format(path, len(dir_files_names_filtered))
    f = open(file_name, "w")
    f.write(dumped_dict)
    f.close()


def init_experiment_env(experiment_dict_, path):
    name = experiment_dict_[EXPERIMENT_NAME]
    print('Initialising an environment for the experiment: {}'.format(name))
    experiment_path = os.path.join(path, name)
    make_dir(name=experiment_path)
    save_experiment_dict(experiment_dict_=experiment_dict_, path=experiment_path)
    return experiment_path


def perform_experiment(experiment_dict_, experiment_path, data_path):
    train_model(experiment_dict_=experiment_dict_, experiment_path=experiment_path, data_path=data_path)


def train_model(experiment_dict_: Dict, experiment_path, data_path):
    dataset_generator = datasets_loading_factory.DatasetGeneratorPyTorch(
        **experiment_dict_[DATASET_HYPERPARAMETERS],
        data_path=data_path
    )
    if experiment_dict_.get(K_HYPERPARAMETERS) is not None:
        k_matrix_factory = derivatives_factory.KMatrixFactory(
            dataset_generator=dataset_generator, **experiment_dict[K_HYPERPARAMETERS]
        )
    else:
        k_matrix_factory = None
    if experiment_dict_.get(H_HYPERPARAMETERS) is not None:
        hessian_factory = derivatives_factory.HessianFactory(
            dataset_generator=dataset_generator, **experiment_dict[H_HYPERPARAMETERS]
        )
    else:
        hessian_factory = None
    model_factory_ = models_factory.ModelsFactory(
        **experiment_dict_[MODEL_HYPERPARAMETERS],
        path_to_save=experiment_path
    )
    optimiser_factory = optimisation_factory.OptimisersFactoryPyTorch(
        **experiment_dict_[OPTIMISER_HYPERPARAMETERS],
        path_to_save=experiment_path
    )
    if experiment_dict_.get(METRICS_HYPERPARAMETERS) is not None:
        metrics_factory_ = metrics_factory.MetricsFactory(**experiment_dict_[METRICS_HYPERPARAMETERS])
    else:
        metrics_factory_ = None
    training_experiment = rnn_training_experiments.RNNTrainingExperimentPyTorch(
        **experiment_dict_[TRAINING_HYPERPARAMETERS],
        experiment_path=experiment_path,
        dataset_generator=dataset_generator,
        models_factory=model_factory_,
        optimisers_factory=optimiser_factory,
        metrics_factory=metrics_factory_,
        k_matrix_factory=k_matrix_factory,
        hessian_factory=hessian_factory,
    )
    training_experiment.run()


if __name__ == '__main__':
    path = input('Please specify global path for experiments: ')
    data_path = os.path.join(path, DATA_PATH)
    experiments_outline = get_experiments_outline()
    print('Performing experiments.')
    for experiment_dict in experiments_outline:
        experiment_path = init_experiment_env(experiment_dict_=experiment_dict, path=path)
        perform_experiment(experiment_dict_=experiment_dict, experiment_path=experiment_path, data_path=data_path)
