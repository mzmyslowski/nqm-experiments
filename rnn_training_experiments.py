import time
from typing import Optional

import numpy as np
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
            k_matrix_factory: derivatives_factory.KMatrixFactory,
            hessian_factory: derivatives_factory.HessianFactory,
            save_best_preds_by: Optional[str] = None,
            best_preds_path: Optional[str] = None,
            eval_metrics_path: Optional[str] = None,
            batch_metrics_path: Optional[str] = None,
            target_metric_value: Optional[float] = None,
            target_metric_name: Optional[str] = None,
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
        self.hessian_factory = hessian_factory
        self.k_matrix_factory = k_matrix_factory
        self.loss_func = None
        self.best_metric_value = None
        self.save_best_preds_by = save_best_preds_by
        self.best_preds_path = best_preds_path
        self.eval_metrics_path = eval_metrics_path
        assert (target_metric_value is None) == (target_metric_name is None)
        self.target_metric_value = target_metric_value
        self.target_metric_name = target_metric_name

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
                self.models_factory.prepare_model_for_training()

                if self.do_cuda():
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                y_batch_pred = self.models_factory.model(x_batch)

                loss = self.loss_func(y_batch_pred, y_batch)

                self._update_weights(loss=loss)

                eval_metrics = {self.BATCH_LOSS: loss.item()}

                if self._do_compute_training_stats(batch_index=batch_index):
                    eval_metrics, preds = self._evaluate_metrics(eval_metrics=eval_metrics)
                    # if self.best_preds_path is not None:
                    #     self._save_best_preds(preds=preds, eval_metrics=eval_metrics)
                    self._print_metrics(eval_metrics=eval_metrics, epoch_index=epoch_index, batch_index=batch_index)
                    if self.eval_metrics_path is not None:
                        self._save_evaluated_metrics(eval_metrics=eval_metrics, epoch_index=epoch_index,
                                                     batch_index=batch_index)
                    self._save_best_model(eval_metrics=eval_metrics)
                    if self._do_stop_training(eval_metrics=eval_metrics):
                        break

            print('Epoch time', time.time() - start)
            if self._do_stop_training(eval_metrics=eval_metrics):
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

    def _compute_training_stats(self, eval_metrics, epoch_index, batch_index):
        # if self.best_preds_path is not None:
        #     self._save_best_preds(preds=preds, eval_metrics=eval_metrics)
        self._print_metrics(eval_metrics=eval_metrics, epoch_index=epoch_index, batch_index=batch_index)
        if self.eval_metrics_path is not None:
            self._save_evaluated_metrics(eval_metrics=eval_metrics, epoch_index=epoch_index, batch_index=batch_index)
        self._save_best_model(eval_metrics=eval_metrics)
        return eval_metrics

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

        print('Computing K')
        k_eigens = self.k_matrix_factory.compute_eigens(model=self.models_factory.model, criterion=self.loss_func)
        print('Computing H')
        h_eigens = self.hessian_factory.compute_eigens(model=self.models_factory.model, criterion=self.loss_func)

        eval_metrics[self.K_EIGENS_TAG] = k_eigens
        eval_metrics[self.H_EIGENS_TAG] = h_eigens

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

    def _save_evaluated_metrics(self, eval_metrics, epoch_index, batch_index):
        np.save(self.eval_metrics_path.format(epoch_index, batch_index), eval_metrics)
        # df = pd.DataFrame(eval_metrics, index=[0])
        # df[self.EPOCH_TAG] = epoch_index
        # df[self.BATCH_TAG] = batch_index
        # if not os.path.exists(self.eval_metrics_path):
        #     df.to_csv(self.eval_metrics_path)
        # else:
        #     with open(self.eval_metrics_path, 'a') as f:
        #         df.to_csv(f, header=False)

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

    def _do_stop_training(self, eval_metrics):
        return (
                eval_metrics.get(self.target_metric_name) is not None and
                eval_metrics[self.target_metric_name] <= self.target_metric_value
        )
