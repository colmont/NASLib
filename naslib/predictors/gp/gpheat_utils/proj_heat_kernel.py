import collections
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

    def __init__(self, sigma=3, kappa=0.05, n_approx=50):
        self.sigma = torch.tensor(sigma, dtype=torch.float32, requires_grad=True)
        self.kappa = torch.tensor(kappa, dtype=torch.float32, requires_grad=True)
        self.n_approx = n_approx
        self.cached = False
        self.all_diff_bits = None
        self.permutations = None

    def fit_transform(self, X, y=None):
        if not isinstance(X, collections.Iterable):
            raise TypeError("Input must be an iterable.")
        # Input validation and parsing
        if not self.cached:
            self.all_diff_bits = self._process_input(X)
            self.cached = True
        return self._compute_kernel(self.all_diff_bits)

    def _process_input(self, X):
        graph_list = self._create_new_graphs(X)
        graph_array = torch.stack(graph_list)
        self._generate_permutations()
        return self._compute_all_diff_bits(graph_array)

    def _create_new_graphs(self, X):
        n, m = 3, 7
        return [self._create_new_graph(graph, n, m) for graph in X]

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
        new_graph = torch.zeros(num_nodes + 2, num_nodes + 2, dtype=torch.int8)

        for node in old_graph[0].keys():
            mapped_node = mapping[node]

            for neighbor, weight in old_graph[0][node].items():
                mapped_neighbor = mapping[neighbor]
                new_graph[mapped_node, mapped_neighbor] = weight

        return new_graph

    def _generate_permutations(self):
        groups = [
            [0],
            [1],
            [2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22],
        ]
        if self.permutations is None:
            self.permutations = torch.tensor(
                self._sample_permutations(groups, self.n_approx)
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

    def _compute_all_diff_bits(self, graph_array):
        perm_len = len(self.permutations)
        all_diff_bits = torch.zeros(
            perm_len * perm_len,
            graph_array.shape[0],
            graph_array.shape[0],
            dtype=torch.int8,
        )

        for i in tqdm(range(perm_len)):
            x1 = graph_array[:, :, self.permutations[i]][:, self.permutations[i], :]

            for j in range(i + 1, perm_len):
                x2 = graph_array[:, :, self.permutations[j]][:, self.permutations[j], :]

                all_diff_bits[i * perm_len + j, :, :] = (
                    x1[:, None].bitwise_xor(x2[None]).sum(dim=(-1, -2))
                )
                all_diff_bits[j * perm_len + i, :, :] = (
                    x1[:, None].bitwise_xor(x2[None]).sum(dim=(-1, -2))
                )

        return all_diff_bits

    def _compute_kernel(self, all_diff_bits):
        kernel = torch.zeros(
            all_diff_bits.shape[1], all_diff_bits.shape[2], requires_grad=True
        )
        for i in range(all_diff_bits.shape[0]):
            kernel = kernel + torch.square(self.sigma) * (
                torch.tanh((torch.square(self.kappa)) / 2) ** all_diff_bits[i]
            )
        kernel /= all_diff_bits.shape[0]
        return kernel
