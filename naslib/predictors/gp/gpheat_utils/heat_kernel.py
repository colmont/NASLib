import collections
import copy
import numpy as np
import torch
from tqdm import tqdm


class HeatKernel:
    """
    HeatKernel class is used for Gaussian process regression.

    Projected heat kernel as described in Borovitskiy et al. (2023). This implementation uses a bit-wise operation
    for efficient computation and also makes use of caching for re-use of similarity computations.

    Attributes
    ----------
    sigma : torch.Tensor
        A hyperparameter in the kernel function.
    kappa : torch.Tensor
        Another hyperparameter in the kernel function.
    n_approx : int
        Number of permutations to be generated for graph comparisons.
    cached : bool
        A boolean flag to indicate if the all_diff_bits have been computed and cached.
    all_diff_bits : torch.Tensor
        Tensor storing the difference in bits between graphs. Used for efficient computation of the kernel.
    permutations : torch.Tensor
        Tensor storing the permutations of graphs to be compared.
    """

    def __init__(self, sigma=3, kappa=0.05, n_approx=50, ss_type='nasbench101', projected=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sigma = torch.tensor(sigma, dtype=torch.float64, requires_grad=True, device=self.device)
        self.kappa = torch.tensor(kappa, dtype=torch.float64, requires_grad=True, device=self.device)
        self.n_approx = n_approx
        self.ss_type = ss_type
        self.cached = False
        self.all_diff_bits = None
        self.permutations = None
        self.n = None
        self.m = None
        self.projected = projected

    def fit_transform(self, X1, X2, y=None):
        if not isinstance(X1, collections.Iterable) or not isinstance(X2, collections.Iterable):
            raise TypeError("Input must be an iterable.")
        X1_copy = copy.deepcopy(X1)
        X2_copy = copy.deepcopy(X2)
        if self.ss_type == 'nasbench101':
            self.n = 3
            self.m = 7
        elif self.ss_type == 'nasbench201':
            self.n = 5
            self.m = 6
            X1_copy = [self._change_nodelabels_nb201(graph) for graph in X1_copy]
            X2_copy = [self._change_nodelabels_nb201(graph) for graph in X2_copy]

        # Input validation and parsing
        if not self.cached:
            self.all_diff_bits = self._process_input(X1_copy, X2_copy)
            self.cached = True

        return self._compute_kernel(self.all_diff_bits)

    def _change_nodelabels_nb201(self, nb201_graph):
        OPS = ["input", "output", "avg_pool_3x3", "nor_conv_1x1", "nor_conv_3x3", "none", "skip_connect"]
        for node in nb201_graph[1].keys():
            nb201_graph[1][node] = OPS.index(nb201_graph[1][node])
        return nb201_graph

    def _process_input(self, X1, X2):
        graph_list_1 = self._create_new_graphs(X1)
        graph_list_2 = self._create_new_graphs(X2)
        graph_array_1 = torch.stack(graph_list_1)
        graph_array_2 = torch.stack(graph_list_2)
        if self.projected:
            self._generate_permutations()
        return self._compute_all_diff_bits(graph_array_1, graph_array_2)

    def _create_new_graphs(self, X):
        return [self._create_new_graph(graph, self.n, self.m) for graph in X]

    def _create_new_graph(self, old_graph, n, m):
        mapping = self._get_mapping(old_graph, n, m)
        return self._map_graph(old_graph, mapping, m * n)

    def _get_mapping(self, old_graph, n, m):
        mapping = {}
        label_counters = np.zeros(n + 2, dtype=int)

        for node in old_graph[0].keys():
            label_index = old_graph[1][node]

            if label_index in [0, 1]:
                mapping[node] = label_index
            else:
                mapping[node] = (
                    label_index * m + label_counters[label_index] - 2 * m + 2
                )
                label_counters[label_index] += 1

        return mapping

    def _map_graph(self, old_graph, mapping, num_nodes):
        new_graph = torch.zeros(num_nodes + 2, num_nodes + 2, dtype=torch.int8, device=self.device)

        for node in old_graph[0].keys():
            mapped_node = mapping[node]

            for neighbor, weight in old_graph[0][node].items():
                mapped_neighbor = mapping[neighbor]
                new_graph[mapped_node, mapped_neighbor] = weight

        return new_graph

    def _generate_permutations(self):
        if self.ss_type == 'nasbench101':
            groups = [
                [0],
                [1],
                [2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22],
            ]
        elif self.ss_type == 'nasbench201':
            groups = [
                [0],
                [1],
                [2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13],
                [14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25],
                [26, 27, 28, 29, 30, 31],
            ]
        if self.permutations is None:
            self.permutations = torch.tensor(
                self._sample_permutations(groups, self.n_approx), device=self.device
            )

    def _sample_permutations(self, groups, x):
        rand_perms = [
            [
                v
                for group in groups
                for v in np.array(group)[np.random.permutation(len(group))]
            ]
            for _ in range(x)
        ]
        return rand_perms

    def _compute_all_diff_bits(self, graph_array_1, graph_array_2):
        if self.projected:
            perm_len = len(self.permutations)
            all_diff_bits = torch.zeros(
                perm_len * perm_len,
                graph_array_1.shape[0],
                graph_array_2.shape[0],
                dtype=torch.int8,
                device=self.device,
            )

            for i in tqdm(range(perm_len)):
                x1 = graph_array_1[:, :, self.permutations[i]][:, self.permutations[i], :]
                y1 = graph_array_2[:, :, self.permutations[i]][:, self.permutations[i], :]

                for j in range(i + 1, perm_len):
                    x2 = graph_array_1[:, :, self.permutations[j]][:, self.permutations[j], :]
                    y2 = graph_array_2[:, :, self.permutations[j]][:, self.permutations[j], :]

                    all_diff_bits[i * perm_len + j, :, :] = (
                        x1[:, None].bitwise_xor(y2[None]).sum(dim=(-1, -2))
                    )
                    all_diff_bits[j * perm_len + i, :, :] = (
                        x2[:, None].bitwise_xor(y1[None]).sum(dim=(-1, -2))
                    )

            return all_diff_bits

        else:
            all_diff_bits = (
                graph_array_1[:, None].bitwise_xor(graph_array_2[None]).sum(dim=(-1, -2))
            )
            return all_diff_bits 
    
    def _compute_kernel(self, all_diff_bits):
        kernel = (
            torch.square(self.sigma) * (torch.tanh(torch.square(self.kappa) / 2) ** all_diff_bits)
        )
        if self.projected:
            kernel = kernel.sum(dim=0)
            kernel /= all_diff_bits.shape[0]
        return kernel.to('cpu')