import collections
import numpy as np
import torch


class HeatKernel:
    """
    HeatKernel class is used for Gaussian process regression.

    Heat kernel as described in Borovitskiy et al. (2023). This implementation uses a bit-wise operation
    for efficient computation and also makes use of caching for re-use of similarity computations.

    Attributes
    ----------
    sigma : torch.Tensor
        A hyperparameter in the kernel function.
    kappa : torch.Tensor
        Another hyperparameter in the kernel function.
    cached : bool
        A boolean flag to indicate if the all_diff_bits have been computed and cached.
    all_diff_bits : torch.Tensor
        Tensor storing the difference in bits between graphs. Used for efficient computation of the kernel.
    """

    def __init__(self, sigma=3, kappa=0.05):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sigma = torch.tensor(sigma, dtype=torch.float64, requires_grad=True, device=self.device)
        self.kappa = torch.tensor(kappa, dtype=torch.float64, requires_grad=True, device=self.device)
        self.cached = False
        self.all_diff_bits = None

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
        new_graph = torch.zeros(num_nodes + 2, num_nodes + 2, dtype=torch.int8, device=self.device)

        for node in old_graph[0].keys():
            mapped_node = mapping[node]

            for neighbor, weight in old_graph[0][node].items():
                mapped_neighbor = mapping[neighbor]
                new_graph[mapped_node, mapped_neighbor] = weight

        return new_graph

    def _compute_all_diff_bits(self, graph_array):
        all_diff_bits = (
            graph_array[:, None].bitwise_xor(graph_array[None]).sum(dim=(-1, -2))
        )
        return all_diff_bits

    def _compute_kernel(self, all_diff_bits):
        kernel = torch.square(self.sigma) * (
            torch.tanh((torch.square(self.kappa)) / 2) ** all_diff_bits
        )
        return kernel.to('cpu')
