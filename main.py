import argparse
import math
import pickle
import random
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from itertools import combinations
from utils import sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp
import torch
from pathlib import Path
import warnings
from models import GNN
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore")




def reset_array():
    class1_train = []
    class2_train = []
    class1_test = []
    class2_test = []
    train_idx = []
    test_idx = []


def meta_train(model, features, labels_local, train_idx, args,adj,metatestFlag):
    weight_decay=args.weight_decay
    epochs=args.n_epochs
    if metatestFlag==True:
        epochs=epochs*5
        weight_decay=weight_decay/5
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        lossf=torch.nn.CrossEntropyLoss()
        noisesubOutput,noisesub,smoothsubOutput,smoothsub_embedding,proxysub_embedding = model(features,adj, labels_local[train_idx], train_idx,args,True)
        gloss=model.Gloss(smoothsub_embedding[train_idx],proxysub_embedding[train_idx],noisesub, labels_local[train_idx], noisesubOutput)
        loss = lossf(noisesubOutput[train_idx], labels_local[train_idx])
        loss = 4*loss+1*gloss
        if loss.item()>5 or math.isnan(loss.item()):
            break
        else:
            loss.backward()
            optimizer.step()
    return model


def meta_test(model, test_features, test_labels, idx_test,adj,args):
    model.eval()
    with torch.no_grad():
        _,_,_,_,proxyoutput= model(test_features,adj, test_labels[idx_test], idx_test,args,False)
        proxyoutput = proxyoutput[idx_test]
        labels = test_labels[idx_test]
        _, indices = torch.max(proxyoutput, dim=1)
        correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)




def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    train_shot = args.train_shot
    test_shot = args.test_shot
    n_classes = args.way

    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    in_feats = features.shape[1]
    g = DGLGraph(data.graph)


    edgeV=g.adj_sparse('coo')[0].numpy()
    edgeU=g.adj_sparse('coo')[1].numpy()
    adj = np.zeros((g.num_nodes(), g.num_nodes()))
    for i in range(g.num_nodes()):
        adj[edgeV[i]][edgeU[i]]=1


    curPath=os.path.abspath(os.path.join(os.getcwd(), ".."))
    NoiseData = Path(curPath+"\\"+args.dataset+"\\"+str(args.Kchange)+"TargetAdj.txt")
    if NoiseData.exists():
        fr = open(curPath+"\\"+args.dataset+"\\"+str(args.Kchange)+"TargetAdj.txt", "rb")
        adj = pickle.load(fr)
        fr.close()
        fr = open(curPath+"\\"+args.dataset+"\\"+str(args.Kchange)+"TargetFea.txt", "rb")
        features = pickle.load(fr)
        fr.close()



    adj = sp.coo_matrix(adj,
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g.to('cuda')
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        print("use cuda:", args.gpu)


    step = args.step
    mean_accuracy_meta_test = []

    if args.dataset == 'cora':
        node_num = 2708
        class_label = [0, 1, 2, 3, 4, 5, 6]
        combination = list(combinations(class_label, n_classes))
    elif args.dataset == 'citeseer':
        node_num = 3327
        class_label = [0, 1, 2, 3, 4, 5]
        combination = list(combinations(class_label, n_classes))



    for i in range(int(len(combination))):
        print('Cross_Validation: ',i+1)
        test_label = list(combination[i])
        train_label = [n for n in class_label if n not in test_label]
        print('Cross_Validation {} Train_Label_List {}: '.format(i + 1, train_label))
        print('Cross_Validation {} Test_Label_List {}: '.format(i + 1, test_label))
        labels_local = labels.clone().detach()
        select_class =random.sample(train_label, n_classes)
        print('Cross_Validation {} Train_Label: {}'.format(i+1, select_class))
        class1_idx = []
        class2_idx = []
        for k in range(node_num):
            if(labels_local[k] == select_class[0]):
                class1_idx.append(k)
                labels_local[k] = 0
            elif(labels_local[k] == select_class[1]):
                class2_idx.append(k)
                labels_local[k] = 1
        model = GNN(g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type).cuda()
        query_idx=[]
        for m in range(step):
            class1_train = random.sample(class1_idx, train_shot)
            class2_train =random.sample(class2_idx, train_shot)
            class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
            train_idx = class1_train + class2_train
            random.shuffle(train_idx)
            test_idx = class1_test + class2_test
            model = meta_train(model, features, labels_local, train_idx, args,adj,False)
            query_idx=query_idx+test_idx
            reset_array()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay*5000)
        model.train()
        optimizer.zero_grad()
        lossf=torch.nn.CrossEntropyLoss()
        noisesubOutput,noisesub,smoothsubOutput,smoothsub_embedding,proxysub_embedding = model(features,adj, labels_local[query_idx], query_idx,args,True)

        gloss=model.Gloss(smoothsub_embedding[query_idx],proxysub_embedding[query_idx],noisesub, labels_local[query_idx], noisesubOutput)
        loss = lossf(noisesubOutput[query_idx], labels_local[query_idx])
        if args.train_shot==5:
            loss = loss+gloss
        else:
            loss = 4*loss+gloss
        loss.backward()
        optimizer.step()

        accuracy_meta_test = []
        torch.save(model.state_dict(), 'model.pkl')

        labels_local = labels.clone().detach()
        select_class = random.sample(test_label, 2)
        class1_idx = []
        class2_idx = []
        reset_array()
        for k in range(node_num):
            if (labels_local[k] == select_class[0]):
                class1_idx.append(k)
                labels_local[k] = 0
            elif (labels_local[k] == select_class[1]):
                class2_idx.append(k)
                labels_local[k] = 1
        for m in range(step):
            class1_train = random.sample(class1_idx, test_shot)
            class2_train = random.sample(class2_idx, test_shot)
            class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
            train_idx = class1_train + class2_train
            random.shuffle(train_idx)
            test_idx = class1_test + class2_test
            model_meta_trained = GNN(g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type).cuda()
            model_meta_trained.load_state_dict(torch.load('model.pkl'))
            model_meta_trained = meta_train(model_meta_trained, features, labels_local, train_idx, args,adj,True)
            acc_test = meta_test(model_meta_trained, features, labels_local, test_idx,adj, args)
            accuracy_meta_test.append(acc_test)
            reset_array()

        print('Cross_Validation: {},batches acc:{}'.format(i + 1,accuracy_meta_test))
        print('*********Meta-Test_Mean Accuracy: {}*******'.format(np.mean(accuracy_meta_test)))
        mean_accuracy_meta_test.append(np.mean(accuracy_meta_test))
        accuracy_meta_test = []


    print('Meta-Test-Mean-Accuracy: {}'.format(torch.tensor(mean_accuracy_meta_test).numpy().mean()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)

    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=5,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=10,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Weight for L2 loss")
    parser.add_argument('--train_shot', type=int, default=5, help='How many shot during meta-train')
    parser.add_argument('--test_shot', type=int, default=5, help='How many shot during meta-test')
    parser.add_argument('--way', type=int, default=2, help='How many shot during meta-test')
    parser.add_argument('--step', type=int, default=5, help='How many times to random select node to test')
    parser.add_argument('--Kchange', type=float, default=20,
                        help='Perturbation rate')
    parser.add_argument('--seed', type=int, default=12, help='Random seed.')
    args = parser.parse_args()
    args.dataset = 'cora'
    main(args)





