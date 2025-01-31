# Author: Xingchen Wan & Binxin Ru @ University of Oxford
# Ru, B., Wan, X., et al., 2021. "Interpretable Neural Architecture Search via Bayesian Optimisation using Weisfeiler-Lehman Kernels". In ICLR 2021.


from copy import deepcopy

from grakel.utils import graph_from_networkx
from termcolor import colored

from naslib.predictors.gp import BaseGPModel
from naslib.predictors.gp.gpwl_utils.convert import *
from naslib.predictors.gp.gpheat_utils.heat_kernel import HeatKernel as Heat

import torch
from torch.nn import Module, Parameter

class ConstrainedParameter(Module):
    def __init__(self, init_value, min_value, max_value):
        super().__init__()
        
        self.min_value = min_value
        self.max_value = max_value
        
        # Initialize the raw parameter, which will be optimized
        self.raw_param = Parameter(torch.tensor(init_value, dtype=torch.float64))
        
    def forward(self):
        # Apply sigmoid to the raw parameter to get it in the range (0, 1)
        sigmoid_param = torch.sigmoid(self.raw_param)
        
        # Scale and shift to the desired range
        constrained_param = self.min_value + (self.max_value - self.min_value) * sigmoid_param
        
        return constrained_param


def _normalize(y):
    y_mean = torch.mean(y)
    y_std = torch.std(y)
    y = (y - y_mean) / y_std
    return y, y_mean, y_std


def _transform(y):
    """By default naslib returns target in terms of accuracy in percentages. We transform this into
    log (error) in decimal"""
    return np.log(1.0 - np.array(y) / 100.0)


def _untransform(y):
    """Inverse operation of _transform(y)"""
    return 100.0 * (1.0 - np.exp(y))


def unnormalize_y(y, y_mean, y_std, scale_std=False):
    """Similar to the undoing of the pre-processing step above, but on the output predictions"""
    if not scale_std:
        y = y * y_std + y_mean
    else:
        y *= y_std
    return y


def _compute_pd_inverse(K, jitter=1e-5):
    """Compute the inverse of a postive-(semi)definite matrix K using Cholesky inversion.
    Return both the inverse matrix and the log determinant."""
    n = K.shape[0]
    # assert isinstance(jitter, float) or jitter.ndim == 0, 'only homoscedastic noise variance is allowed here!'
    is_successful = False
    fail_count = 0
    max_fail = 3
    if not torch.allclose(K, K.t()):
        print("Final K is not symmetric!")
    while fail_count < max_fail and not is_successful:
        try:
            jitter_diag = jitter * torch.eye(n, device=K.device) * 10**fail_count
            K_ = K + jitter_diag
            Kc = torch.linalg.cholesky(K_)
            is_successful = True
        except RuntimeError:
            fail_count += 1
    if not is_successful:
        print(K)
        raise RuntimeError("Gram matrix not positive definite despite of jitter")
    logDetK = -2 * torch.sum(torch.log(torch.diag(Kc)))
    K_i = torch.cholesky_inverse(Kc)
    print("fail count: ", fail_count)
    return K_i, logDetK


def _compute_log_marginal_likelihood(
    K_i,
    logDetK,
    y,
    normalize=True,
):
    """Compute the zero mean Gaussian process log marginal likelihood given the inverse of Gram matrix K(x2,x2), its
    log determinant, and the training label vector y.
    Option:

    normalize: normalize the log marginal likelihood by the length of the label vector, as per the gpytorch
    routine.
    """
    # print(K_i.device, logDetK.device, y.device)
    lml = (
        -0.5 * y.t() @ K_i @ y
        + 0.5 * logDetK
        - y.shape[0]
        / 2.0
        * torch.log(
            2
            * torch.tensor(
                np.pi,
            )
        )
    )
    return lml / y.shape[0] if normalize else lml


class GraphGP:
    def __init__(
        self,
        xtrain,
        ytrain,
        space="nasbench101",
        noise_var=1e-3,
        num_steps=200,
        max_noise_var=1e-1,
        optimize_noise_var=True,
        node_label="op_name",
        projected=False,
        sigma=3.0,
        kappa=0.05,
        n_approx=10,
    ):
        self.likelihood = noise_var
        self.space = space
        self.projected = projected
        self.sigma = sigma
        self.kappa = kappa
        self.n_approx = n_approx

        self.gkernel = None
        # only applicable for the DARTS search space, where we optimise two graphs jointly.
        self.gkernel_reduce = None

        # sometimes (especially for NAS-Bench-201), we can have invalid graphs with all nodes being pruned. Remove
        # these graphs at training time.
        if self.space == "nasbench301" or self.space == "darts":
            # For NAS-Bench-301 or DARTS search space, we need to search for 2 cells (normal and reduction simultaneously)
            valid_indices = [
                i
                for i in range(len(xtrain[0]))
                if len(xtrain[0][i]) and len(xtrain[1][i])
            ]
            self.x = np.array(xtrain)[:, valid_indices]
            # self.x = [xtrain[i] for i in valid_indices]
            self.xtrain_converted = [
                list(
                    graph_from_networkx(
                        self.x[0],
                        node_label,
                    )
                ),
                list(
                    graph_from_networkx(
                        self.x[1],
                        node_label,
                    )
                ),
            ]

        else:
            valid_indices = [i for i in range(len(xtrain)) if len(xtrain[i])]
            temp_list = [xtrain[i] for i in valid_indices]
            # hacky solution for this issue:
            # https://stackoverflow.com/questions/75107506/numpy-wrongly-converts-networkx-graphs-to-array
            self.x = np.empty(len(temp_list), dtype=object)
            for i in range(len(temp_list)):
                self.x[i] = temp_list[i]
            self.xtrain_converted = list(
                graph_from_networkx(
                    self.x,
                    node_label,
                )
            )

        ytrain = np.array(ytrain)[valid_indices]
        self.y_ = deepcopy(
            torch.tensor(ytrain, dtype=torch.float64),
        )
        self.y, self.y_mean, self.y_std = _normalize(deepcopy(self.y_))
        # number of steps of training
        self.num_steps = num_steps

        # other hyperparameters
        self.max_noise_var = max_noise_var
        self.optimize_noise_var = optimize_noise_var

        self.node_label = node_label
        self.K_i = None

    def forward(
        self,
        Xnew,
        full_cov=False,
    ):
        self.gkernel.cached = False

        if self.K_i is None:
            raise ValueError("The GraphGP model has not been fit!")

        # At testing time, similarly we first inspect to see whether there are invalid graphs
        if self.space == "nasbench301" or self.space == "darts":
            invalid_indices = [
                i
                for i in range(len(Xnew[0]))
                if len(Xnew[0][i]) == 0 or len(Xnew[1][i]) == 0
            ]
        else:
            nnodes = np.array([len(x) for x in Xnew])
            invalid_indices = np.argwhere(nnodes == 0)

        # replace the invalid indices with something valid
        patience = 100
        for i in range(len(Xnew)):
            if i in invalid_indices:
                patience -= 1
                continue
            break
        if patience < 0:
            # All architectures are invalid!
            return torch.zeros(len(Xnew)), torch.zeros(len(Xnew))
        for j in invalid_indices:
            if self.space == "nasbench301" or self.space == "darts":
                Xnew[0][int(j)] = Xnew[0][i]
                Xnew[1][int(j)] = Xnew[1][i]
            else:
                Xnew[int(j)] = Xnew[i]

        if self.space == "nasbench301" or self.space == "darts":
            Xnew_T = np.array(Xnew)
            Xnew = np.array(
                [
                    list(
                        graph_from_networkx(
                            Xnew_T[0],
                            self.node_label,
                        )
                    ),
                    list(
                        graph_from_networkx(
                            Xnew_T[1],
                            self.node_label,
                        )
                    ),
                ]
            )

            X_full = np.concatenate((np.array(self.xtrain_converted), Xnew), axis=1)
            K_full = torch.tensor(
                0.5
                * torch.tensor(
                    self.gkernel.fit_transform(X_full[0], X_full[0]), dtype=torch.float64
                )
                + 0.5
                * torch.tensor(
                    self.gkernel_reduce.fit_transform(X_full[1], X_full[1]), dtype=torch.float64
                )
            )
            # Kriging equations
            K_s = K_full[: len(self.x[0]) :, len(self.x[0]) :]
            K_ss = K_full[
                len(self.x[0]) :, len(self.x[0]) :
            ] + self.likelihood * torch.eye(
                Xnew.shape[1],
            )
        else:
            Xnew = list(
                graph_from_networkx(
                    Xnew,
                    self.node_label,
                )
            )
            # Kriging equations
            K_full = self.gkernel.fit_transform(self.xtrain_converted + Xnew, Xnew)
            K_s = K_full[: len(self.xtrain_converted), :]
            K_ss = K_full[len(self.xtrain_converted) :, :] + self.likelihood * torch.eye(len(Xnew))

        mu_s = K_s.t() @ self.K_i @ self.y
        cov_s = K_ss - K_s.t() @ self.K_i @ K_s
        cov_s = torch.clamp(cov_s, self.likelihood, np.inf)
        mu_s = unnormalize_y(mu_s, self.y_mean, self.y_std)
        std_s = torch.sqrt(cov_s)
        std_s = unnormalize_y(std_s, None, self.y_std, True)
        cov_s = std_s**2
        if not full_cov:
            cov_s = torch.diag(cov_s)
        # replace the invalid architectures with zeros
        mu_s[torch.tensor(invalid_indices, dtype=torch.long)] = torch.tensor(
            0.0, dtype=torch.float64
        )
        cov_s[torch.tensor(invalid_indices, dtype=torch.long)] = torch.tensor(
            0.0, dtype=torch.float64
        )
        return mu_s, cov_s

    def fit(self):
        xtrain_grakel = self.xtrain_converted

        self.gkernel = Heat(
            sigma=self.sigma, kappa=self.kappa, n_approx=self.n_approx, ss_type=self.space, projected=self.projected
        )
        self.gkernel_reduce = Heat(
            sigma=self.sigma, kappa=self.kappa, n_approx=self.n_approx, ss_type=self.space, projected=self.projected
        )

        if self.space == "nasbench301" or self.space == "darts":
            K = (
                torch.tensor(
                    self.gkernel.fit_transform(xtrain_grakel[0], xtrain_grakel[0], self.y),
                    dtype=torch.float64,
                )
                + torch.tensor(
                    self.gkernel_reduce.fit_transform(xtrain_grakel[1], xtrain_grakel[1], self.y),
                    dtype=torch.float64,
                )
            ) / 2.0
        else:
            K = self.gkernel.fit_transform(xtrain_grakel, xtrain_grakel, self.y)

        # Here we optimise the noise as a hyperparameter using standard
        # gradient-based optimisation. Here by default we use Adam optimizer.
        if self.optimize_noise_var:
            likelihood = ConstrainedParameter(self.likelihood, 1e-7, self.max_noise_var)
            optim = torch.optim.Adam(
                [likelihood.raw_param, self.gkernel.sigma, self.gkernel.kappa], lr=0.01
            )
            for i in range(self.num_steps):
                optim.zero_grad()
                K = self.gkernel.fit_transform(xtrain_grakel, xtrain_grakel, self.y)
                K_i, logDetK = _compute_pd_inverse(K, likelihood())
                nlml = -_compute_log_marginal_likelihood(K_i, logDetK, self.y)
                nlml.backward()
                print(f"NLML: {nlml.item()}")
                print(f"Gradient of sigma: {self.gkernel.sigma.grad}")
                print(f"Gradient of kappa: {self.gkernel.kappa.grad}")
                optim.step()
                print(colored("likelihood: ", "red"), likelihood().item())
                print(colored("sigma: ", "red"), self.gkernel.sigma.item())
                print(colored("kappa: ", "red"), self.gkernel.kappa.item())
                print()
            # finally
            K_i, logDetK = _compute_pd_inverse(K, likelihood())
            self.K_i = K_i.detach().cpu()
            self.logDetK = logDetK.detach().cpu()
            self.likelihood = likelihood().item()
        else:
            # Compute the inverse covariance matrix
            self.K_i, self.logDetK = _compute_pd_inverse(K, self.likelihood)
            nlml = -_compute_log_marginal_likelihood(self.K_i, self.logDetK, self.y)
            print(colored("NLML: ", "red"), nlml)
        return nlml.item()


class GPHeatPredictor(BaseGPModel):
    def __init__(
        self,
        ss_type="nasbench201",
        optimize_gp_hyper=False,
        num_steps=200,
        projected=False,
        sigma=3.0,
        kappa=0.05,
        n_approx=10,
    ):
        super(GPHeatPredictor, self).__init__(
            encoding_type=None,
            ss_type=ss_type,
            kernel_type=None,
            optimize_gp_hyper=optimize_gp_hyper,
        )
        self.num_steps = num_steps
        self.need_separate_hpo = True
        self.model = None
        self.projected = projected
        self.sigma = sigma
        self.kappa = kappa
        self.n_approx = n_approx

    def _convert_data(self, data: list):
        if self.ss_type == "nasbench101":
            converted_data = [convert_n101_arch_to_graph(arch) for arch in data]
        elif self.ss_type == "nasbench201":
            converted_data = [convert_n201_arch_to_graph(arch) for arch in data]
        elif self.ss_type == "nasbench301" or self.ss_type == "darts":
            converted_data = [convert_darts_arch_to_graph(arch) for arch in data]
            # the converted data is in shape of (N, 2). Transpose to (2,N) for convenience later on.
            converted_data = np.array(converted_data).T.tolist()
        else:
            raise NotImplementedError(
                "Search space %s is not implemented!" % self.ss_type
            )
        return converted_data

    def get_model(self, train_data, **kwargs):
        X_train, y_train = train_data
        # log-transform
        y_train = _transform(y_train)
        # first convert data to networkx
        X_train = self._convert_data(X_train)
        self.model = GraphGP(
            X_train,
            y_train,
            num_steps=self.num_steps,
            optimize_noise_var=self.optimize_gp_hyper,
            space=self.ss_type,
            projected=self.projected,
            sigma=self.sigma,
            kappa=self.kappa,
            n_approx=self.n_approx,
        )
        # fit the model
        self.model.fit()
        return self.model

    def predict(self, input_data, **kwargs):
        X_test = self._convert_data(input_data)
        mean, cov = self.model.forward(X_test, full_cov=True)
        mean = mean.cpu().detach().numpy()
        cov = cov.cpu().detach().numpy()
        mean = _untransform(mean)
        return mean, cov

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        xtrain_conv = self._convert_data(xtrain)
        ytrain_transformed = _transform(ytrain)

        self.model = GraphGP(
            xtrain_conv,
            ytrain_transformed,
            num_steps=self.num_steps,
            optimize_noise_var=self.optimize_gp_hyper,
            space=self.ss_type,
            projected=self.projected,
            sigma=self.sigma,
            kappa=self.kappa,
            n_approx=self.n_approx,
        )
        # fit the model
        nlml = self.model.fit()
        print("Finished fitting GP")
        # predict on training data for checking
        train_pred = self.query(xtrain).squeeze()
        train_error = np.mean(abs(train_pred - ytrain))

        if 'return_nlml' in kwargs:
            return train_error, nlml
        else:
            return train_error

    def query(self, xtest, info=None):
        """alias for predict"""
        mean, cov = self.predict(xtest)
        return mean

    def query_with_cov(self, xtest, info=None):
        mean, cov = self.predict(xtest)
        return mean, cov