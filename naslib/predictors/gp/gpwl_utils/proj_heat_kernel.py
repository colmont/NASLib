"""The weisfeiler lehman kernel :cite:`shervashidze2011weisfeiler`."""
# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
# (Rather extensively) modified by Xingchen Wan <xwan@robots.ox.ac.uk>


import collections
from itertools import permutations, product
from math import tanh
import random
import warnings

import numpy as np
from termcolor import colored
import torch
from tqdm import tqdm

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
        
        self.sigma2 = torch.tensor(3, dtype=torch.float32, requires_grad=True) #FIXME
        self.kappa2 = torch.tensor(0.05, dtype=torch.float32, requires_grad=True) #FIXME
        # self.sigma2 = torch.tensor(0.7, dtype=torch.float32, requires_grad=True) #FIXME
        # self.kappa2 = torch.tensor(1.1, dtype=torch.float32, requires_grad=True) #FIXME
        
        self.cached = False
        self.all_diff_bits = None
        self.random = True
        self.permutations = None
            

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
            
            if self.cached == False:
            
                graph_list = X
                
                def mapping_dic(old_graph, n, m):
                    
                    mapping = {}
                    label_counters = np.zeros(n+2, dtype=int)
                    
                    for node in old_graph[0].keys():
                        label_index = old_graph[1][node]

                        if label_index in [0, 1]: # Nodes with labels 0 and 1 are mapped to single nodes
                            mapping[node] = label_index
                        else:  # Other nodes are mapped according to the original rule
                            mapping[node] = label_index * m + label_counters[label_index] -2*m + 2
                            label_counters[label_index] += 1
                        
                    return mapping

                def create_new_graph(old_graph, n, m):

                    mapping = mapping_dic(old_graph, n, m)
                    
                    # Step 1: Initialize a new graph with m*n nodes
                    new_graph = torch.zeros(m*n+2, m*n+2, dtype=torch.int8)
                    
                    # Step 2: Iterate through each node of the old graph
                    for node in old_graph[0].keys():

                        # Map it to a specific section in the new graph
                        mapped_node = mapping[node]

                        # Step 3: Connect the mapped nodes in the new graph
                        # the same way they were connected in the old graph
                        for neighbor, weight in old_graph[0][node].items():
                            # Get the mapped neighbor
                            mapped_neighbor = mapping[neighbor]
                            # Connect the mapped_node and mapped_neighbor in new_graph
                            new_graph[mapped_node, mapped_neighbor] = weight

                    return new_graph

                        
                # Define your parameters
                n, m = 3, 7
                x = 10
                
                graph_list = [create_new_graph(graph, n, m) for graph in graph_list]
                graph_array = torch.stack(graph_list)
                
                def generate_groups(n, m):
                    return [list(range(i * m, (i + 1) * m)) for i in range(n)]

                def sample_permutations(groups, x):
                    rand_perms = [
                        [v for group in groups for v in np.array(
                            group)[np.random.permutation(len(group))]]
                        for _ in range(x)
                    ]
                    return rand_perms
                
                def generate_permutations(n, m):
                    groups = generate_groups(n, m)
                    perms = [permutations(group, r=len(group)) for group in groups]
                    all_permutations = list(product(*perms))
                    return all_permutations


                # Generate all permutations
                groups = [[0], [1], [2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22]]
                
                
                if self.random == True:
                    sampled_permutations = torch.tensor(sample_permutations(groups, x))
                    self.permutations = sampled_permutations
                    self.random = False
                else:
                    sampled_permutations = self.permutations
                # sampled_permutations = generate_permutations(n, m)

                # @torch.jit.script
                def compute_all_diff_bits(graph_array, sampled_permutations):

                    perm_len = len(sampled_permutations)
                    all_diff_bits = torch.zeros(perm_len*perm_len, graph_array.shape[0], graph_array.shape[0], dtype=torch.int8)                         

                    for i in tqdm(range(len(sampled_permutations))): #tqdm
                        x1 = graph_array[:, :, sampled_permutations[i]][:, sampled_permutations[i], :]
                        
                        for j in range(i+1, len(sampled_permutations)):
                            x2 = graph_array[:, :, sampled_permutations[j]][:, sampled_permutations[j], :]
                
                            all_diff_bits[i * perm_len + j, :, :] = x1[:, None].bitwise_xor(x2[None]).sum(dim=(-1, -2)) #FIXME: // 2
                            all_diff_bits[j * perm_len + i, :, :] = x1[:, None].bitwise_xor(x2[None]).sum(dim=(-1, -2)) #FIXME: // 2

                    return all_diff_bits
                
                def compute_kernel(all_diff_bits):
                    # Initialize the kernel
                    kernel = torch.zeros(graph_array.shape[0], graph_array.shape[0], requires_grad=True)
                    
                    # Compute the kernel
                    for i in range(len(sampled_permutations)**2):
                        kernel = kernel + torch.square(self.sigma2) * (torch.tanh((torch.square(self.kappa2))/2)**all_diff_bits[i])
                    
                    # Normalize the kernel
                    kernel = kernel / (len(sampled_permutations)**2)

                    return kernel

                all_diff_bits = compute_all_diff_bits(graph_array, sampled_permutations)
                K = compute_kernel(all_diff_bits)
                self.all_diff_bits = all_diff_bits
                self.cached = True

            else:

                all_diff_bits = self.all_diff_bits
                
                def compute_kernel(all_diff_bits):
                    # Initialize the kernel
                    kernel = torch.zeros(all_diff_bits.shape[1], all_diff_bits.shape[2], requires_grad=True)
                    
                    # Compute the kernel
                    for i in range(all_diff_bits.shape[0]):
                        kernel = kernel + torch.square(self.sigma2) * (torch.tanh((torch.square(self.kappa2))/2)**all_diff_bits[i])

                    # Normalize the kernel
                    kernel = kernel / (all_diff_bits.shape[0])

                    return kernel

                K = compute_kernel(all_diff_bits)

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
