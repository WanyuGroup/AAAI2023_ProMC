import torch
import torch.nn as nn
import abc
import torch.nn.functional as F
import dgl.function as fn
import numpy as np


class Aggregator(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation=None, bias=True):
        super(Aggregator, self).__init__()
        self.g = g
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, EF) or (2F, EF)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, node):
        nei = node.mailbox['m']  # (B, N, F)
        h = node.data['h']  # (B, F)
        h = self.concat(h, nei, node)  # (B, F) or (B, 2F)
        h = self.linear(h)   # (B, EF)
        if self.activation:
            h = self.activation(h)
        norm = torch.pow(h, 2)
        norm = torch.sum(norm, 1, keepdim=True)
        norm = torch.pow(norm, -0.5)
        norm[torch.isinf(norm)] = 0
        # h = h * norm
        return {'h': h}

    @abc.abstractmethod
    def concat(self, h, nei, nodes):
        raise NotImplementedError

class MeanAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):
        super(MeanAggregator, self).__init__(g, in_feats, out_feats, activation, bias)

    def concat(self, h, nei, nodes):
        nns=nodes.nodes().cpu()
        degs = self.g.in_degrees(nns).float().cuda()
        if h.is_cuda:
            degs = degs.cuda(h.device)
        concatenate = torch.cat((nei, h.unsqueeze(1)), 1)
        concatenate = torch.sum(concatenate, 1) / degs.unsqueeze(1)
        return concatenate  # (B, F)

class PoolingAggregator(Aggregator):
    def __init__(self, g, in_feats, out_feats, activation, bias):  # (2F, F)
        super(PoolingAggregator, self).__init__(g, in_feats*2, out_feats, activation, bias)
        self.mlp = PoolingAggregator.MLP(in_feats, in_feats, F.relu, False, True)

    def concat(self, h, nei, nodes):
        nei = self.mlp(nei)  # (B, F)
        concatenate = torch.cat((nei, h), 1)  # (B, 2F)
        return concatenate

    class MLP(nn.Module):
        def __init__(self, in_feats, out_feats, activation, dropout, bias):  # (F, F)
            super(PoolingAggregator.MLP, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats, bias=bias)  # (F, F)
            self.dropout = nn.Dropout(p=dropout)
            self.activation = activation
            nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        def forward(self, nei):
            nei = self.dropout(nei)  # (B, N, F)
            nei = self.linear(nei)
            if self.activation:
                nei = self.activation(nei)
            max_value = torch.max(nei, dim=1)[0]  # (B, F)
            return max_value

class GNNLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 aggregator_type,
                 bias=True,
                 ):
        super(GNNLayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(p=dropout)
        if aggregator_type == "pooling":
            self.aggregator = PoolingAggregator(g, in_feats, out_feats, activation, bias)
        else:
            self.aggregator = MeanAggregator(g, in_feats, out_feats, activation, bias)

    def forward(self, h):
        h = self.dropout(h)
        self=self.cuda()
        self.g=self.g.to('cuda')
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'), self.aggregator)
        h = self.g.ndata.pop('h')
        return h

class MaskVector(nn.Module):
    def __init__(self, hop):
        super(MaskVector, self).__init__()
        self.hop = hop
        self.weight = torch.nn.Parameter(torch.ones(len(hop), 1))

    def forward(self, gcn_features,rawX):
        h = gcn_features[self.hop].detach()
        x=rawX[self.hop].detach()
        w = torch.sigmoid(self.weight)
        w=w.cuda()
        out = w * h
        out = out.mean(0)
        initalproxySub=w*x
        initalproxySub=initalproxySub.mean(0)
        return out,initalproxySub




class SubG(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes):
        super(SubG, self).__init__()
        self.encoder1 = nn.Linear(in_feats, n_hidden)
        self.decoder1= nn.Linear(n_hidden,n_classes)
        self.fw=nn.Linear(2,2)

    def forward(self, features):
        features=features
        x=self.encoder1(features)
        x = F.relu(x)
        x=self.decoder1(x)
        x = F.relu(x)
        x=self.fw(x)
        return x

    def loss(self,pro,no,sm,idx):
        no=no[idx]
        return torch.sum(F.log_softmax(pro, dim=0)-F.log_softmax(no, dim=0)+F.log_softmax(pro, dim=0)-F.log_softmax(sm, dim=0),dim=0)


class GNN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.cudable=True
        self.n_class=n_classes
        self.in_feats=in_feats

        # input layer
        self.layers.append(GNNLayer(g, in_feats, n_hidden, activation, dropout, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GNNLayer(g, n_hidden, n_hidden, activation, dropout, aggregator_type))
        # output layer
        self.layers.append(GNNLayer(g, n_hidden, n_classes, None, dropout, aggregator_type))
        self.smoSubFeatureMean = torch.zeros(n_classes, n_classes).cuda()
        self.noiseSubFeatureMean =torch.zeros(n_classes, n_classes).cuda()
        self.proxySubFeatureMean = torch.zeros(n_classes, n_classes).cuda()


    def forward(self, features,adj,train_labels,train_idx,args,notevaluateFlag):
        noisesub = features
        for layer in self.layers:
            noisesub = layer(noisesub)

        self.idx_train=train_idx
        smsub=noisesub.clone()
        prosubb=noisesub.clone()
        noisearry=noisesub[np.array(list(range(features.shape[0])))]
        for idx in range(int(len(self.idx_train))):
            node = self.idx_train[idx]
            hop_arr = self.findNhopNodes(adj, node, 1)
            if len(hop_arr)>0:
                noisesub[idx]=torch.sum(noisesub[hop_arr],dim=0)/(len(hop_arr))
                MV = MaskVector(hop_arr)
                Sgenerator=SubG(self.in_feats,5,2).cuda()
                optimizer = torch.optim.SGD(MV.parameters(), lr=0.001)
                optimizergw = torch.optim.SGD(Sgenerator.parameters(), lr=0.001)
                for iter in range(5):
                    if notevaluateFlag:
                        optimizer.zero_grad()
                    smoothsub,initalproxySub = MV(noisesub,features)
                    loss_f = nn.KLDivLoss(size_average=True, reduce=True)
                    MIloss = loss_f(smoothsub.softmax(dim=-1).log().unsqueeze(dim=0),noisearry[idx].softmax(dim=-1).detach().unsqueeze(dim=0))
                    if notevaluateFlag:
                        MIloss.backward()
                        optimizer.step()
                if ~torch.isnan(smoothsub).any():
                    smsub[idx]=smoothsub

                if args.dataset == 'cora':
                    epoch=5
                elif args.dataset == 'citeseer':
                    epoch=3


                for iter in range(epoch):
                    if notevaluateFlag:
                        optimizergw.zero_grad()
                    proxySub=Sgenerator(initalproxySub.detach())
                    loss_fcn = torch.nn.CrossEntropyLoss()
                    classLoss=loss_fcn(F.log_softmax(proxySub, dim=0), train_labels[idx])
                    diffLoss=Sgenerator.loss(proxySub,noisesub.detach(),smoothsub.detach(),idx)
                    if args.train_shot ==5 and args.dataset == 'citeseer':
                        totalloss=2*classLoss+0.5*diffLoss
                    else:
                        totalloss=classLoss+diffLoss
                    if notevaluateFlag:
                        totalloss.backward()
                        optimizergw.step()
                if ~torch.isnan(proxySub).any():
                    prosubb[idx]=proxySub
            smOutput=F.log_softmax(smsub, dim=1)
            proOutput=F.log_softmax(prosubb, dim=1)
            noisesubOutput=F.log_softmax(noisesub, dim=1)
            return noisesubOutput,noisesub,smOutput.detach(),smsub.detach(),proOutput.detach()



    def findNhopNodes(self, adj, node, hop_num):
        def bfs(graph, n, hop_num):
            ans = []
            queue = [n]
            hop_vis = [0]
            seen = set()
            seen.add(n)
            while len(queue) > 0:
                hop = hop_vis.pop(0)
                if hop == hop_num + 1:
                    break
                vertex = queue.pop(0)
                nodes = np.where(graph[vertex].cpu() != 0)[0]
                for w in nodes:
                    if w not in seen:
                        queue.append(w)
                        hop_vis.append(hop + 1)
                        seen.add(w)
                if hop != 0:
                    ans.append(vertex)
            return ans

        if adj.is_sparse:
            adjtemp = adj.to_dense().cpu()
        else:
            adjtemp = adj
        adjtemp[adjtemp != 0] = 1
        ans = bfs(adjtemp, node, hop_num)
        return ans



    def Gloss(self,smoothEmb,proxyEmb,noiseEmb,labels,noiseLabels):
        MSEloss =  nn.MSELoss()
        noiseLabels = torch.max(F.softmax(noiseLabels, dim=1),1)[1]

        proxyEmb = torch.zeros(self.n_class, proxyEmb.shape[1]).cuda().scatter_add(0, torch.transpose(labels.repeat(smoothEmb.shape[1], 1), 1, 0), proxyEmb)
        proxyEmb = torch.div(proxyEmb, torch.max(torch.zeros(self.n_class).cuda().scatter_add(0, labels, torch.ones_like(labels, dtype=torch.float).cuda()),  torch.ones_like(torch.zeros(self.n_class).cuda().scatter_add(0, labels, torch.ones_like(labels, dtype=torch.float).cuda())).cuda()).view(self.n_class, 1))
        noiseEmb = torch.zeros(self.n_class, smoothEmb.shape[1]).cuda().scatter_add(0, torch.transpose(noiseLabels.repeat(smoothEmb.shape[1], 1), 1, 0), noiseEmb)
        noiseEmb = torch.div(noiseEmb, torch.max(torch.zeros(self.n_class).cuda().scatter_add(0, noiseLabels, torch.ones_like(noiseLabels, dtype=torch.float).cuda()), torch.ones_like(torch.zeros(self.n_class).cuda().scatter_add(0, noiseLabels, torch.ones_like(noiseLabels, dtype=torch.float).cuda())).cuda()).view(self.n_class, 1))
        smoothEmb = torch.zeros(self.n_class, smoothEmb.shape[1]).cuda().scatter_add(0, torch.transpose(labels.repeat(smoothEmb.shape[1], 1), 1, 0), smoothEmb)
        smoothEmb = torch.div(smoothEmb, torch.max(torch.zeros(self.n_class).cuda().scatter_add(0, labels, torch.ones_like(labels, dtype=torch.float).cuda()),  torch.ones_like(torch.zeros(self.n_class).cuda().scatter_add(0, labels, torch.ones_like(labels, dtype=torch.float).cuda())).cuda()).view(self.n_class, 1))

        return MSEloss(smoothEmb/2, noiseEmb/2)+MSEloss(proxyEmb/6, noiseEmb/6)




