from typing import Dict, Optional
import torch


class OptimisersFactory:

    def __init__(self):
        self.optimiser = None

    def init_optimiser(self, weights):
        raise NotImplementedError

    def save_optimiser(self):
        raise NotImplementedError


class OptimisersFactoryPyTorch(OptimisersFactory):
    OPTIMISERS = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop,
    }

    LR = 'lr'
    L2_RATE = 'weight_decay'

    def __init__(
            self,
            optimiser_name: str,
            lr: float,
            l2_rate: float,
            additional_params: Optional[Dict] = None,
            optimiser_path: Optional[str] = None,
            use_new_params: bool = True,
            path_to_save: str = './optimiser.pt'
    ):
        super().__init__()
        self.optimiser_name = optimiser_name
        self.lr = lr
        self.l2_rate = l2_rate
        self.additional_params = additional_params if additional_params is not None else {}
        self.optimiser_path = optimiser_path
        self.use_new_params = use_new_params
        self.path_to_save = path_to_save

    def init_optimiser(self, weights):
        optimiser = self.OPTIMISERS[self.optimiser_name](
            params=weights,
            lr=self.lr,
            weight_decay=self.l2_rate,
            **self.additional_params
        )
        if self.optimiser_path is not None:
            self._load_optimiser_params(optimiser=optimiser)
            if self.use_new_params:
                for p in optimiser.param_groups:
                    p[self.LR] = self.lr
                    p[self.L2_RATE] = self.l2_rate
                    for k, v in self.additional_params.items():
                        p[k] = v
        self.optimiser = optimiser
        return optimiser

    def _load_optimiser_params(self, optimiser):
        checkpoint = torch.load(self.optimiser_path)
        optimiser.load_state_dict(checkpoint)

    def save_optimiser(self):
        torch.save(self.optimiser.state_dict(), self.path_to_save)


class SchedulersFactory:

    def __init__(self):
        self.scheduler = None

    def init_scheduler(self, optimiser):
        raise NotImplementedError

    def scheduler_step(self, metric=None):
        raise NotImplementedError

    def save_scheduler(self):
        raise NotImplementedError


class SchedulersFactoryPyTorch(SchedulersFactory):
    SCHEDULERS = {
        'Reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'Cyclical': torch.optim.lr_scheduler.CyclicLR
    }
    REDUCE = 'Reduce'

    def __init__(
            self,
            scheduler_name: str,
            scheduler_params: Optional[Dict] = None,
            scheduler_path: Optional[str] = None,
            path_to_save: str = './scheduler.pt',
    ):
        super().__init__()
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params if scheduler_params is not None else {}
        self.scheduler_path = scheduler_path
        self.path_to_save = path_to_save

    @property
    def use_metric(self):
        return self.scheduler_name == self.REDUCE

    def init_scheduler(self, optimiser):
        scheduler = self.SCHEDULERS[self.scheduler_name](optimizer=optimiser, **self.scheduler_params)
        if self.scheduler_path is not None:
            self._load_scheduler_params(scheduler=scheduler)
        self.scheduler = scheduler
        return scheduler

    def scheduler_step(self, metric=None):
        if self.scheduler is not None:
            if self.use_metric:
                self.scheduler.step(metrics=metric)
            else:
                self.scheduler.step()

    def _load_scheduler_params(self, scheduler):
        checkpoint = torch.load(self.scheduler_path)
        scheduler.load_state_dict(checkpoint)

    def save_scheduler(self):
        torch.save(self.scheduler.state_dict(), self.path_to_save)

