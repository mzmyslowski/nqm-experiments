import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

import datasets_loading_factory
import metrics_factory
import models_factory
import optimisation_factory


class RNNTrainingExperimentPyTorch:
    LOSSES = {
        'CrossEntropy': torch.nn.CrossEntropyLoss
    }
    LOSS_NAME = 'LOSS'
    TRAIN_NAME = 'TRAIN'
    TEST_NAME = 'TEST'
    PREDS_NAME = 'Y_PREDS'
    Y_NAME = 'Y'

    def __init__(
            self,
            loss_name: str,
            epochs: int,
            compute_training_stats_step: int,
            training_batch_size: int,
            predict_batch_size: int,
            random_seed: int,
            dataset_generator: datasets_loading_factory.DatasetGeneratorPyTorch,
            models_factory: models_factory.ModelsFactory,
            optimisers_factory: optimisation_factory.OptimisersFactory,
            metrics_factory: metrics_factory.MetricsFactory,
            save_best_preds_by: Optional[str] = None,
            best_preds_path: Optional[str] = None,
            eval_metrics_path: Optional[str] = None,
    ):
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
        self.loss_func = None
        self.best_metric_value = None
        self.save_best_preds_by = save_best_preds_by
        self.best_preds_path = best_preds_path
        self.eval_metrics_path = eval_metrics_path

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
        data_loader = self.dataset_generator.get_data_loader(batch_size=self.training_batch_size)
        eval_metrics = {}
        for epoch_index in range(self.epochs):
            start = time.time()
            for batch_index, (x_batch, y_batch) in enumerate(data_loader):
                self.models_factory.prepare_model_for_training()

                if self.do_cuda():
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                y_batch_pred = self.models_factory.model(x_batch)

                loss = self.loss_func(y_batch_pred, y_batch)

                self._update_weights(loss=loss)

                if self._do_compute_training_stats(batch_index=batch_index):
                    eval_metrics = self._compute_training_stats(epoch_index=epoch_index, batch_index=batch_index)

            print('Epoch time', time.time() - start)
            if eval_metrics['TRAIN_LOSS'] <= 0.01:
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

    def _compute_training_stats(self, epoch_index, batch_index):
        eval_metrics, train_preds = self._evaluate_metrics(prefix_name=self.TRAIN_NAME)
        # if self.best_preds_path is not None:
        #     self._save_best_preds(preds=preds, eval_metrics=eval_metrics)
        self._print_metrics(eval_metrics=eval_metrics, epoch_index=epoch_index, batch_index=batch_index)
        if self.eval_metrics_path is not None:
            self._save_evaluated_metrics(eval_metrics=eval_metrics)
        self._save_best_model(eval_metrics=eval_metrics)
        return eval_metrics

    def _evaluate_metrics(self, prefix_name):
        self._prepare_model_for_testing()

        eval_metrics = {}

        y_pred, y_true = self._get_all_preds_batchwise()

        eval_metrics['{}_{}'.format(prefix_name, self.LOSS_NAME)] = self.loss_func(y_pred, y_true).item()

        preds_dict = {
            '{}_{}'.format(prefix_name, self.PREDS_NAME): y_pred,
            '{}_{}'.format(prefix_name, self.Y_NAME): y_true,
        }

        # eval_metrics = self.metrics_factory.evaluate_metrics(
        #     y=y_true,
        #     y_pred=y_pred,
        #     prefix_name=prefix_name,
        # )
        return eval_metrics, preds_dict

    def _prepare_model_for_testing(self):
        self.models_factory.prepare_model_for_testing()

    def _save_best_preds(self, preds, eval_metrics):
        if (
                self.save_best_preds_by is not None and
                (
                        self.best_metric_value is None or
                        self.best_metric_value >= eval_metrics[self.save_best_preds_by]
                )
        ):
                self.best_metric_value = eval_metrics[self.save_best_preds_by]
                file_name = self.best_preds_path
                np.save(file_name, preds)

    def _print_metrics(self, eval_metrics, epoch_index, batch_index):
        printing_index = 'Epoch {} Batch {}'.format(epoch_index, batch_index)
        for metric_name, metric_value in eval_metrics.items():
            print(printing_index, metric_name, metric_value)
        print('\n')

    def _save_evaluated_metrics(self, eval_metrics):
        df = pd.DataFrame(eval_metrics, index=[0])
        if not os.path.exists(self.eval_metrics_path):
            df.to_csv(self.eval_metrics_path)
        else:
            with open(self.eval_metrics_path, 'a') as f:
                df.to_csv(f, header=False)

    def _save_best_model(self, eval_metrics):
        if (
                self.save_best_preds_by is not None and
                (
                        self.best_metric_value is None or
                        self.best_metric_value >= eval_metrics[self.save_best_preds_by]
                )
        ):
            self.best_metric_value = eval_metrics[self.save_best_preds_by]
            self.models_factory.save_model()
            self.optimisers_factory.save_optimiser()

    def _get_all_preds_batchwise(self):
        y_pred = []
        y_true = []
        data_loader = self.dataset_generator.get_data_loader(batch_size=self.predict_batch_size)
        for index, (x_batch, y_batch) in enumerate(data_loader):
            if self.do_cuda():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            y_pred_batch = self.models_factory.model(x_batch).data
            y_pred.append(y_pred_batch)
            y_true.append(y_batch)
        y_pred_tensor = torch.cat(y_pred)
        y_true_tensor = torch.cat(y_true)
        return y_pred_tensor, y_true_tensor