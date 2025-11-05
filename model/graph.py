import numpy as np
import torch
import scipy.sparse as sp
import ot
import networkx as nx
from torch_sparse import SparseTensor
from scipy import stats
from scipy.spatial import distance
from sklearn import metrics
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors, kneighbors_graph


class graph:
    def __init__(self, data=None, adata=None, k=6, dmax=50, rad_cutoff=150,
                 distType='BallTree', mode='KNN', method='knn'):
        """
        data: numpy array (for other method)
        adata: AnnData object (for knn method)
        method: 'knn' or 'other'
        """
        self.data = data
        self.adata = adata
        self.k = k
        self.dmax = dmax
        self.mode = mode
        self.rad_cutoff = rad_cutoff
        self.distType = distType
        self.method = method

    # ---------- knn MODE ----------
    def _generate_adj_knn(self):
        position = self.adata.obsm['spatial']
        distance_matrix = ot.dist(position, position, metric='euclidean')
        n_spot = distance_matrix.shape[0]
        interaction = np.zeros([n_spot, n_spot])
        for i in range(n_spot):
            vec = distance_matrix[i, :]
            distance = vec.argsort()
            for t in range(1, self.k + 1):
                y = distance[t]
                interaction[i, y] = 1
        adj = interaction + interaction.T
        adj = np.where(adj > 1, 1, adj)
        return adj

    def _generate_adj_dist(self):
        dist = metrics.pairwise_distances(self.adata.obsm['spatial'], metric='euclidean')
        adj_mat = (dist < self.dmax).astype(np.int64)
        return adj_mat

    @staticmethod
    def _sparse_mx_to_torch_sparse_tensor(sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def _preprocess_graph(self, adj):
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self._sparse_mx_to_torch_sparse_tensor(adj_normalized)

    def _knn_method(self):
        if self.mode == 'KNN':
            adj = self._generate_adj_knn()
        else:
            adj = self._generate_adj_dist()
        adj = sp.coo_matrix(adj)
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        adj_norm = self._preprocess_graph(adj)
        adj = adj + sp.eye(adj.shape[0])
        adj = adj.tocoo()
        shape = adj.shape
        values = adj.data
        indices = np.stack([adj.row, adj.col])
        adj_label = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        norm_value = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        graph_dict = {
            "adj_norm": adj_norm,
            "adj_label": adj_label,
            "norm_value": norm_value
        }
        return graph_dict

    # ---------- other MODE ----------
    def _graph_computing(self):
        dist_list = ["euclidean","braycurtis","canberra","mahalanobis","chebyshev","cosine",
                     "jensenshannon","minkowski","seuclidean","sqeuclidean","hamming",
                     "jaccard","kulsinski","matching","rogerstanimoto","russellrao",
                     "sokalmichener","sokalsneath","wminkowski","yule"]
        graphList = []
        if self.distType == 'spearmanr':
            SpearA, _ = stats.spearmanr(self.data, axis=1)
            for node_idx in range(self.data.shape[0]):
                tmp = SpearA[node_idx, :].reshape(1, -1)
                res = tmp.argsort()[0][-(self.k+1):]
                for j in np.arange(0, self.k):
                    graphList.append((node_idx, res[j]))

        elif self.distType == "BallTree":
            tree = BallTree(self.data)
            _, ind = tree.query(self.data, k=self.k+1)
            indices = ind[:, 1:]
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))

        elif self.distType == "KDTree":
            tree = KDTree(self.data)
            _, ind = tree.query(self.data, k=self.k+1)
            indices = ind[:, 1:]
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))

        elif self.distType == "kneighbors_graph":
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity', include_self=False)
            A = A.toarray()
            for node_idx in range(self.data.shape[0]):
                indices = np.where(A[node_idx] == 1)[0]
                for j in np.arange(0, len(indices)):
                    graphList.append((node_idx, indices[j]))

        elif self.distType == "Radius":
            nbrs = NearestNeighbors(radius=self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            for node_idx in range(indices.shape[0]):
                for j in range(indices[node_idx].shape[0]):
                    if distances[node_idx][j] > 0:
                        graphList.append((node_idx, indices[node_idx][j]))

        elif self.distType in dist_list:
            for node_idx in range(self.data.shape[0]):
                tmp = self.data[node_idx, :].reshape(1, -1)
                distMat = distance.cdist(tmp, self.data, self.distType)
                res = distMat.argsort()[:self.k + 1]
                tmpdist = distMat[0, res[0][1:self.k + 1]]
                boundary = np.mean(tmpdist) + np.std(tmpdist)
                for j in np.arange(1, self.k+1):
                    if distMat[0, res[0][j]] <= boundary:
                        graphList.append((node_idx, res[0][j]))
        else:
            raise ValueError(f"{self.distType!r} is not supported.")
        return graphList

    def _List2Dict(self, graphList):
        graphdict = {}
        tdict = {}
        for g in graphList:
            end1, end2 = g
            tdict[end1] = ""
            tdict[end2] = ""
            if end1 in graphdict:
                tmplist = graphdict[end1]
            else:
                tmplist = []
            tmplist.append(end2)
            graphdict[end1] = tmplist
        for i in range(self.data.shape[0]):
            if i not in tdict:
                graphdict[i] = []
        return graphdict

    def _mx2SparseTensor(self, mx):
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).to(torch.long)
        col = torch.from_numpy(mx.col).to(torch.long)
        values = torch.from_numpy(mx.data)
        adj = SparseTensor(row=row, col=col, value=values, sparse_sizes=mx.shape)
        return adj.t()

    def _pre_graph(self, adj):
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self._mx2SparseTensor(adj_normalized)

    def _other_method(self):
        adj_mtx = self._graph_computing()
        graphdict = self._List2Dict(adj_mtx)
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
        adj_pre = adj_org - sp.dia_matrix((adj_org.diagonal()[np.newaxis, :], [0]), shape=adj_org.shape)
        adj_pre.eliminate_zeros()
        adj_norm = self._pre_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)
        graph_dict = {
            "adj_norm": adj_norm,
            "adj_label": adj_label,
            "norm_value": norm
        }
        return graph_dict
    
    def main(self):
        if self.method == 'knn':
            return self._knn_method()
        elif self.method == 'other':
            return self._other_method()
        else:
            raise ValueError("method must be 'knn' or 'other'")
