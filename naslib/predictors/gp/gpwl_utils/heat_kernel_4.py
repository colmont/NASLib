"""The weisfeiler lehman kernel :cite:`shervashidze2011weisfeiler`."""
# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
# (Rather extensively) modified by Xingchen Wan <xwan@robots.ox.ac.uk>


import collections
from itertools import permutations, product
from math import tanh
import warnings

import numpy as np
from termcolor import colored
import torch

from naslib.predictors.gp.gpwl_utils.vertex_histogram import CustomVertexHistogram
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from grakel.graph import Graph
from grakel.kernels import Kernel

# Python 2/3 cross-compatibility import
from six import iteritems
from six import itervalues
from copy import deepcopy

from collections import OrderedDict

import networkx as nx


class Heat(Kernel):
    """Compute the Weisfeiler Lehman Kernel.

     See :cite:`shervashidze2011weisfeiler`.

    Parameters
    ----------
    h : int, default=5
        The number of iterations.

    base_graph_kernel : `grakel.kernels.Kernel` or tuple, default=None
        If tuple it must consist of a valid kernel object and a
        dictionary of parameters. General parameters concerning
        normalization, concurrency, .. will be ignored, and the
        ones of given on `__init__` will be passed in case it is needed.
        Default `base_graph_kernel` is `VertexHistogram`.

    node_weights: iterable
        If not None, the nodes will be assigned different weights according
        to this vector. Must be a dictionary with the following format:
        {'node_name1': weight1, 'node_name2': weight2 ... }
        Must be of the same length as the number of different node attributes

    Attributes
    ----------
    X : dict
     Holds a dictionary of fitted subkernel modules for all levels.

    _nx : number
        Holds the number of inputs.

    _h : int
        Holds the number, of iterations.

    _base_graph_kernel : function
        A void function that initializes a base kernel object.

    _inv_labels : dict
        An inverse dictionary, used for relabeling on each iteration.

    """

    _graph_format = "dictionary"

    def __init__(
        self,
        normalize=True,
        h=2,
        base_graph_kernel=CustomVertexHistogram,
    ):
        """Initialise a `weisfeiler_lehman` kernel."""
        super(Heat, self).__init__(
            n_jobs=None, verbose=False, normalize=normalize
        )

        self.h = h
        self.base_graph_kernel = base_graph_kernel
        self._initialized.update({"h": False, "base_graph_kernel": False})
        self._base_graph_kernel = None
        self.feature_dims = [
            0,
        ]  # Record the dimensions of the vectors of each WL iteration
        self.sigma2 = torch.tensor(0.05, dtype=torch.float32, requires_grad=True) #FIXME
        self.kappa2 = torch.tensor(0.05, dtype=torch.float32, requires_grad=True) #FIXME
            

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        super(Heat, self).initialize()
        if not self._initialized["base_graph_kernel"]:
            base_graph_kernel = self.base_graph_kernel
            if base_graph_kernel is None:
                base_graph_kernel, params = CustomVertexHistogram, dict()
            elif type(base_graph_kernel) is type and issubclass(
                base_graph_kernel, Kernel
            ):
                params = dict()
            else:
                try:
                    base_graph_kernel, params = base_graph_kernel
                except Exception:
                    raise TypeError(
                        "Base kernel was not formulated in "
                        "the correct way. "
                        "Check documentation."
                    )

                if not (
                    type(base_graph_kernel) is type
                    and issubclass(base_graph_kernel, Kernel)
                ):
                    raise TypeError(
                        "The first argument must be a valid "
                        "grakel.kernel.kernel Object"
                    )
                if type(params) is not dict:
                    raise ValueError(
                        "If the second argument of base "
                        "kernel exists, it must be a diction"
                        "ary between parameters names and "
                        "values"
                    )
                params.pop("normalize", None)

            params["normalize"] = False
            # params["verbose"] = False
            # params["n_jobs"] = None
            self._base_graph_kernel = base_graph_kernel
            self._params = params
            self._initialized["base_graph_kernel"] = True

        if not self._initialized["h"]:
            if type(self.h) is not int or self.h < 0:
                raise TypeError(
                    "'h' must be a non-negative integer. Got h" + str(self.h)
                )
            self._h = self.h + 1
            self._initialized["h"] = True

    def parse_input(
        self,
        X,
    ):
        """Parse input for weisfeiler lehman.

        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that correspond to the given
            graph format). A valid input also consists of graph type objects.

        return_embedding_only: bool
            Whether to return the embedding of the graphs only, instead of computing the kernel all
            the way to the end.

        Returns
        -------
        base_graph_kernel : object
        Returns base_graph_kernel.

        """
        if self._method_calling not in [1, 2]:
            raise ValueError(
                "method call must be called either from fit " + "or fit-transform"
            )
        elif hasattr(self, "_X_diag"):
            # Clean _X_diag value
            delattr(self, "_X_diag")

        # Input validation and parsing
        if not isinstance(X, collections.Iterable):
            raise TypeError("input must be an iterable\n")
        else:
            graph_list = list()
        
            for x in X:
                graph = nx.DiGraph(x[0])
                nx.set_node_attributes(graph, x[1], 'label')
                graph_list.append(graph)
        
            def initialize_label_counters(n):
                label_counters = {}
                for label_index in range(n):
                    label_counters[label_index] = 0
                return label_counters

            def map_node(node, label_counters, old_graph, m):
                label_index = old_graph.nodes[node]['label']
                mapped_node = label_index * m + label_counters[label_index]
                label_counters[label_index] += 1
                return mapped_node

            def create_new_graph(old_graph, n=6, m=9): #FIXME
                # Step 1: Initialize a new graph with m*n nodes
                new_graph = nx.DiGraph()
                new_graph.add_nodes_from(range(m * n))

                # Initialize a counter for each label
                label_counters = initialize_label_counters(n)

                # Step 2: Iterate through each node of the old graph
                for node in old_graph.nodes():

                    # Map it to a specific section in the new graph
                    mapped_node = map_node(node, label_counters, old_graph, m)

                    # Step 3: Connect the mapped nodes in the new graph
                    # the same way they were connected in the old graph
                    for neighbor in old_graph.neighbors(node):
                        # Get the mapped neighbor
                        mapped_neighbor = map_node(neighbor, label_counters, old_graph, m)
                        # Connect the mapped_node and mapped_neighbor in new_graph
                        new_graph.add_edge(mapped_node, mapped_neighbor)

                return new_graph            

            def differing_bits(graph1, graph2):
                # Convert graphs to adjacency matrices as NumPy arrays with integer data type
                nodelist1 = list(graph1.nodes())
                nodelist1.sort()
                adj_matrix1 = nx.to_numpy_array(graph1, dtype=int, nodelist=nodelist1)
                nodelist2 = list(graph2.nodes())
                nodelist2.sort()
                adj_matrix2 = nx.to_numpy_array(graph2, dtype=int, nodelist=nodelist2)
                
                # Compute the differing bits
                diff_bits = np.bitwise_xor(adj_matrix1, adj_matrix2).sum()

                return int(diff_bits)
            
            def generate_groups(n, m):
                # Generate the numbers from 0 to n*m
                numbers = list(range(n*m))

                # Split the numbers into groups
                groups = [numbers[i:i+m] for i in range(0, len(numbers), m)]

                return groups

            def generate_permutations(n, m):
                # Generate the groups
                groups = generate_groups(n, m)

                # Generate all permutations of each group
                permuted_groups = [list(permutations(group)) for group in groups]

                # Create a list of dictionaries, each representing a permutation
                output = []
                for permutation_product in product(*permuted_groups):
                    perm_dict = {}
                    for original_group, permuted_group in zip(groups, permutation_product):
                        perm_dict.update(dict(zip(original_group, permuted_group)))
                    output.append(perm_dict)

                return output

            # Define your parameters
            n, m = 6, 9

            # Generate all permutations
            all_permutations = generate_permutations(n, m)

            def compute_kernel(graph1, graph2, all_permutations):
                # Initialize the kernel
                kernel = 0

                # Iterate through all permutations
                for permutation_1 in all_permutations:
                    for permutation_2 in all_permutations:
                        
                        # Compute the kernel for the current permutation
                        kernel_permutation = compute_kernel_permutation(graph1, graph2, permutation_1, permutation_2)

                        # Add the kernel for the current permutation to the total kernel
                        kernel += kernel_permutation

                # Normalize the kernel
                kernel /= (len(all_permutations)**2)

                return kernel
            
            def compute_kernel_permutation(graph1, graph2, permutation_1, permutation_2):

                modified_graph1 = nx.relabel_nodes(graph1, permutation_1)
                modified_graph2 = nx.relabel_nodes(graph2, permutation_2)

                # Compute the differing bits
                diff_bits = differing_bits(modified_graph1, modified_graph2)

                # Compute the kernel
                kernel = torch.square(self.sigma2) * (torch.tanh((torch.square(self.kappa2))/2)**diff_bits)

                return kernel

            # initialize the matrix as a torch tensor
            K = torch.empty(len(graph_list), len(graph_list))

            # compute the bitwise xor operation between all pairs
            for i in range(len(graph_list)):
                for j in range(i + 1, len(graph_list)):
                    K_new = compute_kernel(graph_list[i], graph_list[j], all_permutations)
                    K[i, j] = K_new
                    K[j, i] = K_new
            
            # replace diagonal elements with 1
            for i in range(len(graph_list)):
                K[i, i] = torch.square(self.sigma2)*1

        base_graph_kernel = None # empty dummy variable

        if self._method_calling == 2:
            return K, base_graph_kernel

    def fit_transform(self, X, y=None):
        """Fit and transform, on the same dataset.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.

        y : Object, default=None
            Ignored argument, added for the pipeline.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 2
        self._is_transformed = False
        self.initialize()
        self.feature_dims = [
            0,
        ]  # Flush the feature dimensions
        if X is None:
            raise ValueError("transform input cannot be None")
        else:
            km, self.X = self.parse_input(X)

        # print(colored(km, "red"))
        return km

    def transform(
        self,
        X,
    ):
        """Calculate the kernel matrix, between given and fitted dataset.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.

            Whether to return the embedding of the graphs only, instead of computing the kernel all
            the way to the end.
        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 3
        # Check is fit had been called
        check_is_fitted(self, ["X", "_nx", "_inv_labels"])

        # Input validation and parsing
        if X is None:
            raise ValueError("transform input cannot be None")
        else:
            if not isinstance(X, collections.Iterable):
                raise ValueError("input must be an iterable\n")
            else:
                nx = 0
                distinct_values = set()
                Gs_ed, L = dict(), dict()
                for (i, x) in enumerate(iter(X)):
                    is_iter = isinstance(x, collections.Iterable)
                    if is_iter:
                        x = list(x)
                    if is_iter and len(x) in [0, 2, 3]:
                        if len(x) == 0:
                            warnings.warn("Ignoring empty element on index: " + str(i))
                            continue

                        elif len(x) in [2, 3]:
                            x = Graph(x[0], x[1], {}, self._graph_format)
                    elif type(x) is Graph:
                        x.desired_format("dictionary")
                    else:
                        raise ValueError(
                            "each element of X must have at "
                            + "least one and at most 3 elements\n"
                        )
                    Gs_ed[nx] = x.get_edge_dictionary()
                    L[nx] = x.get_labels(purpose="dictionary")

                    # Hold all the distinct values
                    distinct_values |= set(
                        v for v in itervalues(L[nx]) if v not in self._inv_labels[0]
                    )
                    nx += 1
                if nx == 0:
                    raise ValueError("parsed input is empty")

        nl = len(self._inv_labels[0])
        WL_labels_inverse = {
            dv: idx for (idx, dv) in enumerate(sorted(list(distinct_values)), nl)
        }
        WL_labels_inverse = OrderedDict(WL_labels_inverse)

        def generate_graphs_transform(WL_labels_inverse, nl):
            # calculate the kernel matrix for the 0 iteration
            new_graphs = list()
            for j in range(nx):
                new_labels = dict()
                for (k, v) in iteritems(L[j]):
                    if v in self._inv_labels[0]:
                        new_labels[k] = self._inv_labels[0][v]
                    else:
                        new_labels[k] = WL_labels_inverse[v]
                L[j] = new_labels
                # produce the new graphs
                new_graphs.append([Gs_ed[j], new_labels])
            yield new_graphs

            for i in range(1, self._h):
                new_graphs = list()
                L_temp, label_set = dict(), set()
                nl += len(self._inv_labels[i])
                for j in range(nx):
                    # Find unique labels and sort them for both graphs
                    # Keep for each node the temporary
                    L_temp[j] = dict()
                    for v in Gs_ed[j].keys():
                        credential = (
                            str(L[j][v])
                            + ","
                            + str(sorted([L[j][n] for n in Gs_ed[j][v].keys()]))
                        )
                        L_temp[j][v] = credential
                        if credential not in self._inv_labels[i]:
                            label_set.add(credential)

                # Calculate the new label_set
                WL_labels_inverse = dict()
                if len(label_set) > 0:
                    for dv in sorted(list(label_set)):
                        idx = len(WL_labels_inverse) + nl
                        WL_labels_inverse[dv] = idx

                # Recalculate labels
                new_graphs = list()
                for j in range(nx):
                    new_labels = dict()
                    for (k, v) in iteritems(L_temp[j]):
                        if v in self._inv_labels[i]:
                            new_labels[k] = self._inv_labels[i][v]
                        else:
                            new_labels[k] = WL_labels_inverse[v]
                    L[j] = new_labels
                    # Create the new graphs with the new labels.
                    new_graphs.append([Gs_ed[j], new_labels])
                yield new_graphs

        # Calculate the kernel matrix without parallelization
        K = np.sum(
            (
                self.X[i].transform(
                    g,
                    label_start_idx=self.feature_dims[i],
                    label_end_idx=self.feature_dims[i + 1],
                )
                for (i, g) in enumerate(
                    generate_graphs_transform(WL_labels_inverse, nl)
                )
            ),
            axis=0,
        )

        self._is_transformed = True
        if self.normalize:
            X_diag, Y_diag = self.diagonal()
            # if self.as_tensor:
            #     div_ = torch.sqrt(torch.ger(Y_diag, X_diag))
            #     K /= div_
            # else:
            old_settings = np.seterr(divide="ignore")
            K = np.nan_to_num(np.divide(K, np.sqrt(np.outer(Y_diag, X_diag))))
            np.seterr(**old_settings)

        return K

    def diagonal(self):
        """Calculate the kernel matrix diagonal for fitted data.

        A funtion called on transform on a seperate dataset to apply
        normalization on the exterior.

        Parameters
        ----------
        None.

        Returns
        -------
        X_diag : np.array
            The diagonal of the kernel matrix, of the fitted data.
            This consists of kernel calculation for each element with itself.

        Y_diag : np.array
            The diagonal of the kernel matrix, of the transformed data.
            This consists of kernel calculation for each element with itself.

        """
        # Check if fit had been called
        check_is_fitted(self, ["X"])
        try:
            check_is_fitted(self, ["_X_diag"])
            if self._is_transformed:
                Y_diag = self.X[0].diagonal()[1]
                for i in range(1, self._h):
                    Y_diag += self.X[i].diagonal()[1]
        except NotFittedError:
            # Calculate diagonal of X
            if self._is_transformed:
                X_diag, Y_diag = self.X[0].diagonal()
                # X_diag is considered a mutable and should not affect the kernel matrix itself.
                X_diag.flags.writeable = True
                for i in range(1, self._h):
                    x, y = self.X[i].diagonal()
                    X_diag += x
                    Y_diag += y
                    self._X_diag = X_diag
                else:
                    # case sub kernel is only fitted
                    X_diag = self.X[0].diagonal()
                    # X_diag is considered a mutable and should not affect the kernel matrix itself.
                    X_diag.flags.writeable = True
                    for i in range(1, self._n_iter):
                        x = self.X[i].diagonal()
                        X_diag += x
                    self._X_diag = X_diag

        # if self.as_tensor:
        #     self._X_diag = torch.tensor(self._X_diag)
        #     if Y_diag is not None:
        #         Y_diag = torch.tensor(Y_diag)
        if self._is_transformed:
            return self._X_diag, Y_diag
        else:
            return self._X_diag
