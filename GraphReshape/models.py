import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

args_cuda = torch.cuda.is_available()

class GCN(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, type='GCN'):
        super(GCN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.h1 = nn.Linear(x_dim, h_dim)
        self.h2 = nn.Linear(h_dim, y_dim)
        self.type = type


    def forward(self, x, adj):

        if self.type == 'LPGCN':
            if args_cuda:
                adj_ = adj + torch.eye(adj.shape[0]).cuda()
            else:
                adj_ = adj + torch.eye(adj.shape[0])
            rowsum = adj_.sum(dim=1)
            degree_mat_inv_sqrt = torch.diag(rowsum.pow(-0.5))
            adj = adj_.mm(degree_mat_inv_sqrt).transpose(0,1).mm(degree_mat_inv_sqrt)

        h = self.h1(torch.spmm(adj, x))
        h = F.relu(h)
        embedding = torch.spmm(adj, h)
        y = self.h2(embedding)

        return y, embedding

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.h1 = nn.Linear(x_dim, h_dim)

        self.act=nn.PReLU()

    def forward(self, x, adj, neighbor=True):
        if neighbor:
            h = self.h1(torch.spmm(adj, x))
        else:
            h = self.h1(x)

        h = self.act(h)
        h = F.softmax(h)
        return h

class GraphReshape(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(GraphReshape, self).__init__()
        if args_cuda:
            self.embedding = Encoder(x_dim, h_dim).cuda()
        else:
            self.embedding = Encoder(x_dim, h_dim)

    
    def forward(self, x, adj):

        h_node = self.embedding(x, adj, neighbor=False)
        h_graph = self.embedding(x, adj)
        
        return h_node, h_graph
