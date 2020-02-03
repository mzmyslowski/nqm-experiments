from typing import Callable, List, Optional, Union

import numpy as np
import scipy.sparse.linalg
import torch
import torch.nn.functional as F


class KMatrixFactory:
    def __init__(self, dataset_generator, batch_size, L=100, center=True):
        self.dataset_generator = dataset_generator
        self.batch_size = batch_size
        self.L = L
        self.center = center

    def compute_eigens(self, model, criterion):
        gs1 = []
        data_loader = self.dataset_generator.get_data_loader(batch_size=self.batch_size, shuffle=True)
        it = iter(data_loader)
        n = 0
        mean_g = 0
        #for _ in tqdm.tqdm(range(L), total=L):
        for _ in range(self.L):
            inputs, targets = next(it)
            if do_cuda():
                inputs = inputs.cuda()
                targets = targets.cuda()
            output = model(inputs)
            loss = criterion(output, targets)
            grad_dict = torch.autograd.grad(
               loss, model.parameters(), create_graph=True
            )
            g = torch.nn.utils.parameters_to_vector(grad_dict)  # g jest na batchu
            gs1.append(g.detach().cpu().numpy())
            mean_g += gs1[-1] / float(self.L)  # Very important to normalize! Otherwise centering is incorrect.
            n += 1

        G = np.zeros(shape=(n, n))

        for i in range(len(G)):
            for j in range(len(G)):
                if self.center:
                    G[i, j] = (gs1[i] - mean_g).dot(gs1[j] - mean_g)
                else:
                    G[i, j] = (gs1[i]).dot(gs1[j])
        G /= n

        _, eigenvalues, _ = np.linalg.svd(G)

        ids = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[ids]
        return eigenvalues


class HessianFactory:
    def __init__(self, k, dataset_generator, batch_size, sample_percentage, tolerance=1e-2):
        self.k = k
        self.dataset_generator = dataset_generator
        self.batch_size = batch_size
        self.sample_percentage = sample_percentage
        self.tolerance = tolerance

    def compute_eigens(self, model, criterion):
        data_loader, _ = self.dataset_generator.get_random_sampled_data_loader(
            batch_size=self.batch_size,
            sample_percentage=self.sample_percentage
        )
        eigenvalues, eigenvectors = self._get_k_loss_hessian_eigens(
            k=self.k,
            model=model,
            data_loader=data_loader,
            tolerance=self.tolerance,
            criterion=criterion,
        )
        n_samples = self.sample_percentage * len(self.dataset_generator.dataset)
        normalized_eigenvalues = self._normalize_eigenvalues(
            eigenvalues=eigenvalues,
            sample_len=n_samples,
            batch_size=self.batch_size
        )
        return normalized_eigenvalues

    def _get_k_loss_hessian_eigens(
            self,
            k: int,
            model,
            data_loader,
            criterion: Callable = F.cross_entropy,
            tolerance: float = 1e-2,
    ):
        N = sum(p.numel() for p in model.parameters())
        Hv = self._get_Hv_function(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
        )
        Hv_linear_operator = scipy.sparse.linalg.LinearOperator(shape=(N, N), matvec=Hv)
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(Hv_linear_operator, k=k, tol=tolerance)
        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors)
        if do_cuda():
            eigenvalues, eigenvectors = eigenvalues.cuda(), eigenvectors.cuda()
        return eigenvalues, eigenvectors

    def _get_Hv_function(self, model, data_loader, criterion):
        def Hv(v):
            vectors = self._divide_flattened_vector_between_parameters(
                flattened_vector=v,
                model_parameters=model.parameters()
            )
            self._evaluate_hessian_vector_product(
                vectors=vectors,
                model=model,
                data_loader=data_loader,
                criterion=criterion,
                model_parameters=model.parameters()
            )
            result = self._extract_hessian_vector_product_to_vector(model_parameters=model.parameters())
            return result
        return Hv

    def _divide_flattened_vector_between_parameters(self, flattened_vector, model_parameters):
        parameters_vectors = []
        parameters_num_sum = 0
        for parameter in model_parameters:
            parameter_num = parameter.numel()
            vector = flattened_vector[parameters_num_sum:parameters_num_sum+parameter_num]
            parameter_vector = torch.from_numpy(vector).view(parameter.size()).float()
            if do_cuda():
                parameter_vector = parameter_vector.cuda()
            parameters_vectors.append(parameter_vector)
            parameters_num_sum += parameter_num
        return parameters_vectors

    def _evaluate_hessian_vector_product(self, vectors, model, data_loader, criterion, model_parameters):
        model.eval()
        model.zero_grad()
        for batch_idx, (data, target) in enumerate(data_loader):
            if do_cuda():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            grad = self._compute_gradient(
                outputs=loss,
                inputs=model_parameters,
                create_graph=True,
                retain_graph=True,
            )
            hessian_vector_product_sum = torch.zeros(1)
            if do_cuda():
                hessian_vector_product_sum = hessian_vector_product_sum.cuda()
            for index, (g, v) in enumerate(zip(grad, vectors)):
                hessian_vector_product_sum += (g * v).sum()
            hessian_vector_product_sum.backward()

    def _compute_gradient(
            self,
            outputs: torch.Tensor,
            inputs: Union[List[torch.nn.Parameter], torch.Tensor],
            create_graph: bool = False,
            retain_graph: bool = False,
            v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        grads = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=v, create_graph=create_graph,
                                    retain_graph=retain_graph)
        return grads

    def _extract_hessian_vector_product_to_vector(self, model_parameters):
        flattened_gradients = [p.grad.cpu().numpy().ravel() for p in model_parameters]
        concatenated_flattened_gradients = np.concatenate(flattened_gradients)
        return concatenated_flattened_gradients

    def _normalize_eigenvalues(self, eigenvalues, sample_len, batch_size):
        return eigenvalues * batch_size / sample_len


def do_cuda():
    return torch.cuda.device_count() > 0

