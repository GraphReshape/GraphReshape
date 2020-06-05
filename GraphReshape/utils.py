from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    ----------
    adj : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels

def preprocess_graph(adj, I=True):
    """
    Perform the processing of the adjacency matrix proposed by Kipf et al. 2017.

    Parameters
    ----------
    adj: sp.spmatrix
        Input adjacency matrix.

    Returns
    -------
    The matrix (D+1)^(-0.5) (adj + I) (D+1)^(-0.5)

    """
    if I:
        adj_ = adj + sp.eye(adj.shape[0])
    else:
        adj_ = adj
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized

def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None,
                                 random_state=None):

    """
    Split the arrays or matrices into random train, validation and test subsets.
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;
    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.
    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def load_data(name='cora_0.1edges_Meta-Self.npy', dataset='cora.npz', attack=True):

    _A_obs, _X_obs, _z_obs = load_npz('data/'+dataset)
    if _X_obs is None:
        _X_obs = sp.eye(_A_obs.shape[0]).tocsr()

    lcc = largest_connected_components(_A_obs)

    if attack:
        _A_obs = sp.csr_matrix(np.load('data/'+name))
        
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    if not attack:
        _A_obs = _A_obs[lcc][:,lcc]
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()
    _X_obs = _X_obs.astype("float32")

    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"

    _X_obs = _X_obs[lcc]
    _z_obs = _z_obs[lcc]
    _N = _A_obs.shape[0]
    _K = _z_obs.max()+1
    _Z_obs = np.eye(_K)[_z_obs]

    seed = 15
    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share
    np.random.seed(seed)

    split_train, split_val, split_unlabeled = train_val_test_split_tabular(np.arange(_N),
                                                                           train_size=train_share,
                                                                           val_size=val_share,
                                                                           test_size=unlabeled_share,
                                                                           stratify=_z_obs)
    split_unlabeled = np.union1d(split_val, split_unlabeled)
    
    labels = torch.LongTensor(_Z_obs)
    labels = torch.max(labels, dim=1)[1]
    features = torch.FloatTensor(np.array(_X_obs.todense())).float()
    
    args_cuda = torch.cuda.is_available()
    set_seed(42, args_cuda)
    if args_cuda:
        labels = labels.cuda()
        features = features.cuda()
    
    return labels, features, split_train, split_val, split_unlabeled, _A_obs

def plot_with_labels(embedding, labels, dataset, phase):

    tsne = TSNE(n_components=2)
    lowDWeights = tsne.fit_transform(embedding)
    cc = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.cla()
    plt.rcParams['figure.dpi'] = 300
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    i = 0
    for x, y, s in zip(X, Y, labels):
        plt.scatter(x, y, 2, c=cc[s])
        i += 1
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.axis('off')
    plt.title(phase + ' Graph ' + '(' + dataset + ')')
    plt.savefig('result/' + phase + '_' + dataset + '.png', dpi=300)
    plt.show()