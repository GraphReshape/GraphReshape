import torch
import numpy as np
from random import randint, sample
from torch import nn
from torch.nn import functional as F
import copy
from GraphReshape.models import GCN, GraphReshape
from GraphReshape.utils import *
from tqdm import tqdm
import scipy.sparse as sp
args_cuda = torch.cuda.is_available()

def eval_gcn(model, x, adj, num_test, labels, class_num):
    test_size = len(num_test)
    p_y_pred = model(x, adj)[0][num_test]
    pre = p_y_pred.view(test_size, class_num) 
    pre = pre.argmax(dim=1)
    y_ = labels[num_test]

    return float((y_==pre).sum().item()) / test_size

def train_gcn(model, labels, features, adj, epochs, optimizer, num_train, num_val, num_test):
    for _ in range(epochs):
        epoch_loss = 0.0
        
        model.training = True
        optimizer.zero_grad()
        x = features.view(-1, features.size(-1))
        y_pred = model(x, adj)[0][num_train]
        label = labels[num_train]
        y_pred = y_pred.view(-1, labels.max()+1)
        loss = F.cross_entropy(y_pred, label)

        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss.item()

        model.training = False

    acc = eval_gcn(model, x, adj, num_test, labels, labels.max()+1)

    return model, acc

def get_gcn(labels, features, adj, num_train, num_val, num_test, epochs=100, h_dim=50, type='GCN'):
    model = GCN(x_dim=features.shape[-1], h_dim=h_dim, y_dim=int(labels.max() + 1), type=type)
    if args_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
    model, acc = train_gcn(model, labels, features, adj, epochs, optimizer, num_train, num_val, num_test)

    return model, acc

def finetune_gcn(adj, labels, features, split_train, split_val, split_unlabeled, model, epochs=3):    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-6)
    model.training = True
    model, _ = train_gcn(model, labels, features, adj, epochs, optimizer, split_train, split_val, split_unlabeled)
    
    return model

def train_graphreshape(model, features, adj, epochs, optimizer, n_sample=1):
    early_stopping = 400
    best_loss = 100000
    print('abnormal detect.....')
    for _ in range(epochs):
        epoch_loss = 0.0

        model.training = True
        optimizer.zero_grad()
        x = features.view(-1, features.size(-1))
        p_node, p_graph = model(x, adj)

        if args_cuda:
            a = torch.min((torch.abs(torch.cosine_similarity(p_node, p_graph)).mean()), torch.Tensor([0.7]).cuda())
        else:
            a = torch.min((torch.abs(torch.cosine_similarity(p_node, p_graph)).mean()), torch.Tensor([0.7]))

        b = 0
        for _ in range(n_sample):
            idx = np.random.permutation(len(x))
            if args_cuda: 
                b += torch.max((torch.abs(torch.cosine_similarity(p_node[idx], p_graph)).mean()), torch.Tensor([0.2]).cuda())
            else:
                b += torch.max((torch.abs(torch.cosine_similarity(p_node[idx], p_graph)).mean()), torch.Tensor([0.2]))
                
        b = 0 if n_sample == 0 else b / n_sample    
        loss = (1 - a + b)
        
        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss.item()
            
        if epoch_loss < best_loss:
            early_stopping = 400
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = epoch_loss
        else:
            early_stopping -= 1
        
        if early_stopping == 0:
            print('Early stopping!')
            break

    model.load_state_dict(best_model_wts)
    print('-'*50)

    return model

def graphreshape(raw_adj, features, labels, split_train, split_val, split_unlabeled, n_sample=1, lr=5e-4, weight_decay=5e-6, h_dim=50 ,threold=0.7):
    x_dim = features.shape[-1]
    h_dim = h_dim  

    if args_cuda:
        model = GraphReshape(x_dim, h_dim).cuda()
    else:
        model = GraphReshape(x_dim, h_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.training = True

    adj = torch.Tensor(preprocess_graph(raw_adj, I=False).toarray())

    if args_cuda:
        adj = adj.cuda()

    model = train_graphreshape(model, features, adj, 100000, optimizer, n_sample)

    h_node, h_graph = model(features, adj)

    score = torch.cosine_similarity(h_node, h_graph)
    abnormal_node = np.where(score.cpu().detach().numpy() < threold * 0.7 + (1-threold) * 0.2)[0]
    
    abnormal_train = []
    for i in abnormal_node:
        if i in split_train:
            abnormal_train.append(i)
    normal_train = list(set(split_train)-set(abnormal_train)) 
    all_normal = list(set(range(len(adj)))-set(abnormal_train))

    for i in abnormal_train:
        raw_adj[i, :] = 0
        raw_adj[:, i] = 0

    adj = torch.Tensor(raw_adj.toarray()).cuda()

    if len(abnormal_train) == 0:
        print('no abnormal node')
    else:
        GCN_model,_ = get_gcn(labels, features, adj, split_train, split_val, split_unlabeled, epochs=50, h_dim=50, type='LPGCN')

    print('graph reshape........')
    for _ in tqdm(range(int(0.5 * raw_adj.sum()*len(abnormal_train)/len(raw_adj.toarray())))):

        if len(abnormal_train) == 0:
            break

        adj = torch.Tensor(raw_adj.toarray()).cuda() 

        GCN_model.zero_grad()
        GCN_model = finetune_gcn(adj, labels, features, split_train, split_val, split_unlabeled, GCN_model)
        GCN_model.eval()
        GCN_model.zero_grad()
        
        adj.requires_grad=True
        loss = F.cross_entropy(GCN_model(features, adj)[0][normal_train], labels[normal_train]) 
        loss.backward()
        
        grad = adj.grad.cpu().numpy() - 1 + 2 * raw_adj.toarray()
        grad[all_normal] = 0
        grad_add_edge, grad_remove_edge = grad.min(), grad.max()

        op = 1 if abs(grad_add_edge) > grad_remove_edge else 0

        if op == 1:
            edge_x, edge_y = np.where(grad == grad.min())[0][0], np.where(grad == grad.min())[1][0]
        else:
            edge_x, edge_y = np.where(grad == grad.max())[0][0], np.where(grad == grad.max())[1][0]

        raw_adj[edge_x, edge_y] = op
        raw_adj[edge_y, edge_x] = op

    return raw_adj, model

