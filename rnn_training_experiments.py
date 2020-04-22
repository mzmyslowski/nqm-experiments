import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

import derivatives_factory
import datasets_loading_factory
import metrics_factory
import models_factory
import optimisation_factory


class RNNTrainingExperimentPyTorch:
    LOSSES = {
        'CrossEntropy': torch.nn.CrossEntropyLoss
    }
    LOSS_TAG = 'LOSS'
    TRAIN_TAG = 'TRAIN'
    TEST_TAG = 'TEST'
    PREDS_TAG = 'Y_PREDS'
    Y_TAG = 'Y'
    BATCH_LOSS = 'BATCH_LOSS'
    BATCH_TAG = 'batch'
    EPOCH_TAG = 'epoch'
    K_EIGENS_TAG = 'K_EIGENS'
    H_EIGENS_TAG = 'H_EIGENS'
    MEAN_GRADIENT_NORM_TAG = 'MEAN_GRADIENT_NORM'
    WEIGHTS_NORM_TAG = 'WEIGHTS_NORM'
    WEIGHTS_INIT_COSINE_TAG = 'WEIGHTS_INIT_COSINE'
    WEIGHTS_INIT_DISTANCE_TAG = 'WEIGHTS_INIT_DISTANCE'
    GRADIENT_HESSIAN_OVERLAP_TAG = 'GRADIENT_HESSIAN_OVERLAP'
    BEST_PREDS_RELATIVE_PATH = 'best_preds.npy'
    BEST_MODEL_RELATIVE_PATH = 'best_model.pt'
    BEST_OPTIMISER_RELATIVE_PATH = 'best_optimiser.pt'
    MODEL_RELATIVE_PATH = 'model.pt'
    OPTIMISER_RELATIVE_PATH = 'optimiser.pt'

    def __init__(
            self,
            experiment_path: str,
            loss_name: str,
            epochs: int,
            compute_training_stats_step: int,
            training_batch_size: int,
            predict_batch_size: int,
            random_seed: int,
            dataset_generator: datasets_loading_factory.DatasetGeneratorPyTorch,
            models_factory: models_factory.ModelsFactory,
            optimisers_factory: optimisation_factory.OptimisersFactory,
            metrics_factory: Optional[metrics_factory.MetricsFactory] = None,
            k_matrix_factory: Optional[derivatives_factory.KMatrixFactory] = None,
            hessian_factory: Optional[derivatives_factory.HessianFactory] = None,
            save_best_preds_by: Optional[str] = None,
            target_metric_value: Optional[float] = None,
            target_metric_name: Optional[str] = None,
            max_epoch_without_improvement: Optional[int] = None,
            eval_metrics_file_name: str = 'eval_metrics.csv',
            save_model: bool = True,
            save_best_model: bool = True,
    ):
        self.experiment_path = experiment_path
        self.best_preds_path = os.path.join(experiment_path, self.BEST_PREDS_RELATIVE_PATH)
        self.eval_metrics_path = os.path.join(experiment_path, eval_metrics_file_name)
        self.loss_name = loss_name
        self.epochs = epochs
        self.compute_training_stats_step = compute_training_stats_step
        self.training_batch_size = training_batch_size
        self.predict_batch_size = predict_batch_size
        self.random_seed = random_seed
        self.dataset_generator = dataset_generator
        self.models_factory = models_factory
        self.optimisers_factory = optimisers_factory
        self.metrics_factory = metrics_factory
        self.hessian_factory = hessian_factory
        self.k_matrix_factory = k_matrix_factory
        self.loss_func = None
        self.best_metric_value = None
        self.target_metric_value = target_metric_value
        self.target_metric_name = target_metric_name
        self.max_epoch_without_improvement = max_epoch_without_improvement
        self.best_epoch = 0
        self.save_model = save_model
        self.save_best_model = save_best_model

    def run(self):
        self.dataset_generator.init_dataset()
        model = self.models_factory.init_model(
            dimensionality=self.dataset_generator.dimensionality,
            n_channels=self.dataset_generator.n_channels,
            n_classes=self.dataset_generator.n_classes
        )
        self.optimisers_factory.init_optimiser(weights=model.parameters())
        self.loss_func = self._get_loss_func()
        self._train()

    def _get_loss_func(self):
        return self.LOSSES[self.loss_name]()

    def _train(self):
        data_loader = self.dataset_generator.get_data_loader(batch_size=self.training_batch_size, train=True)
        eval_metrics = {}
        for epoch_index in range(self.epochs):
            start = time.time()
            for batch_index, (x_batch, y_batch) in enumerate(data_loader):

                if self._do_compute_training_stats(batch_index=batch_index):
                    eval_metrics, preds = self._evaluate_metrics(eval_metrics=eval_metrics)
                    if self.target_metric_name is not None:
                        self._update_target_metric_value(eval_metrics=eval_metrics, epoch=epoch_index)
                    self._print_metrics(eval_metrics=eval_metrics, epoch_index=epoch_index, batch_index=batch_index)
                    if self.eval_metrics_path is not None:
                        self._save_evaluated_metrics(eval_metrics=eval_metrics, epoch_index=epoch_index,
                                                     batch_index=batch_index)
                    if self.save_model:
                        self._save_model(model_name=self.MODEL_RELATIVE_PATH, optimiser_name=self.OPTIMISER_RELATIVE_PATH)
                    if self.target_metric_name is not None and self.save_best_model:
                        self._save_best_model(eval_metrics=eval_metrics)
                    if self._do_stop_training(eval_metrics=eval_metrics, epoch=epoch_index):
                        break

                self.models_factory.prepare_model_for_training()

                if self.do_cuda():
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                y_batch_pred = self.models_factory.model(x_batch)

                loss = self.loss_func(y_batch_pred, y_batch)

                self._update_weights(loss=loss)

            print('Epoch time', time.time() - start)
            if self._do_stop_training(eval_metrics=eval_metrics, epoch=epoch_index):
                break

    @staticmethod
    def do_cuda():
        return torch.cuda.device_count() > 0

    def _update_weights(self, loss):
        self.models_factory.model.zero_grad()
        loss.backward()
        self.optimisers_factory.optimiser.step()

    def _do_compute_training_stats(self, batch_index):
        return batch_index % self.compute_training_stats_step == 0

    def _evaluate_metrics(self, eval_metrics):
        self._prepare_model_for_testing()

        train_data_loader = self.dataset_generator.get_data_loader(batch_size=self.predict_batch_size, train=True)
        test_data_loader = self.dataset_generator.get_data_loader(batch_size=self.predict_batch_size, train=False)

        print('Computing train predictions')
        y_train_pred, y_train_true = self._get_all_preds_batchwise(data_loader=train_data_loader)
        print('Computing test predictions')
        y_test_pred, y_test_true = self._get_all_preds_batchwise(data_loader=test_data_loader)

        preds_dict = {
            '{}_{}'.format(self.TRAIN_TAG, self.PREDS_TAG): y_train_pred,
            '{}_{}'.format(self.TRAIN_TAG, self.Y_TAG): y_train_true,
            '{}_{}'.format(self.TEST_TAG, self.PREDS_TAG): y_test_pred,
            '{}_{}'.format(self.TEST_TAG, self.Y_TAG): y_test_true,
        }

        print('Computing train loss')
        eval_metrics['{}_{}'.format(self.TRAIN_TAG, self.LOSS_TAG)] = self.loss_func(y_train_pred, y_train_true).item()
        print('Computing test loss')
        eval_metrics['{}_{}'.format(self.TEST_TAG, self.LOSS_TAG)] = self.loss_func(y_test_pred, y_test_true).item()

        if self.k_matrix_factory is not None:
            print('Computing K')
            k_eigens, mean_gradient = self.k_matrix_factory.compute_eigens(
                model=self.models_factory.model, criterion=self.loss_func
            )
            k_eigens_dict = {'{}_{}'.format(self.K_EIGENS_TAG, index): k for index, k in enumerate(k_eigens)}
            eval_metrics.update(k_eigens_dict)
            eval_metrics[self.MEAN_GRADIENT_NORM_TAG] = np.linalg.norm(mean_gradient)

        if self.hessian_factory is not None:
            print('Computing H')
            h_eigenvalues, h_eigenvectors = self.hessian_factory.compute_eigens(
                model=self.models_factory.model, criterion=self.loss_func
            )
            h_eigens_dict = {
                '{}_{}'.format(self.H_EIGENS_TAG, index): h for index, h in enumerate(reversed(h_eigenvalues))
            }
            eval_metrics.update(h_eigens_dict)

        if self.hessian_factory is not None and self.k_matrix_factory is not None:
            eval_metrics[self.GRADIENT_HESSIAN_OVERLAP_TAG] = self._compute_gradient_hessian_overlap(
                gradient=mean_gradient,
                H=h_eigenvectors
            )

        weights = torch.nn.utils.parameters_to_vector(self.models_factory.model.parameters())
        eval_metrics[self.WEIGHTS_NORM_TAG] = weights.norm().item()
        eval_metrics[self.WEIGHTS_INIT_COSINE_TAG] = (
                self.models_factory.init_weights.dot(weights) /
                (self.models_factory.init_weights.norm() * weights.norm())
        ).item()
        eval_metrics[self.WEIGHTS_INIT_DISTANCE_TAG] = (self.models_factory.init_weights - weights).norm().item()
        return eval_metrics, preds_dict

    def _compute_gradient_hessian_overlap(self, gradient, H):
        return np.linalg.norm(gradient.dot(H)) / np.linalg.norm(gradient)

    def _prepare_model_for_testing(self):
        self.models_factory.prepare_model_for_testing()

    def _update_target_metric_value(self, eval_metrics, epoch):
        if (
                self.best_metric_value is None or
                self.best_metric_value > eval_metrics[self.target_metric_name]
        ):
            self.best_metric_value = eval_metrics[self.target_metric_name]
            self.best_epoch = epoch

    # def _save_best_preds(self, preds, eval_metrics):
    #     if (
    #             self.save_best_preds_by is not None and
    #             (
    #                     self.best_metric_value is None or
    #                     self.best_metric_value >= eval_metrics[self.save_best_preds_by]
    #             )
    #     ):
    #             self.best_metric_value = eval_metrics[self.save_best_preds_by]
    #             file_name = self.best_preds_path
    #             np.save(file_name, preds)

    def _print_metrics(self, eval_metrics, epoch_index, batch_index):
        printing_index = 'Epoch {} Batch {}'.format(epoch_index, batch_index)
        for metric_name, metric_value in eval_metrics.items():
            print(printing_index, metric_name, metric_value)
        print('\n')

    def _save_evaluated_metrics(self, eval_metrics, epoch_index, batch_index):
        df = pd.DataFrame(eval_metrics, index=[0])
        df[self.EPOCH_TAG] = epoch_index
        df[self.BATCH_TAG] = batch_index
        if not os.path.exists(self.eval_metrics_path):
            df.to_csv(self.eval_metrics_path)
        else:
            df.to_csv(self.eval_metrics_path, mode='a', header=False)

    def _save_best_model(self, eval_metrics):
        if self.best_metric_value >= eval_metrics[self.target_metric_name]:
            self._save_model(model_name=self.BEST_MODEL_RELATIVE_PATH, optimiser_name=self.BEST_OPTIMISER_RELATIVE_PATH)

    def _save_model(self, model_name, optimiser_name):
        self.models_factory.save_model(name=model_name)
        self.optimisers_factory.save_optimiser(name=optimiser_name)

    def _get_all_preds_batchwise(self, data_loader):
        y_pred = []
        y_true = []
        for index, (x_batch, y_batch) in enumerate(data_loader):
            if self.do_cuda():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            y_pred_batch = self.models_factory.model(x_batch).data
            y_pred.append(y_pred_batch)
            y_true.append(y_batch)
        y_pred_tensor = torch.cat(y_pred)
        y_true_tensor = torch.cat(y_true)
        return y_pred_tensor, y_true_tensor

    def _do_stop_training(self, eval_metrics, epoch):
        return (
                eval_metrics.get(self.target_metric_name) is not None and
                eval_metrics[self.target_metric_name] <= self.target_metric_value
        ) or (
                self.max_epoch_without_improvement is not None and
                self.target_metric_name is not None and
                self.max_epoch_without_improvement <= epoch - self.best_epoch
        )
