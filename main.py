import torch
import torch_geometric as torchg
from libs import datap, plott
import torch_geometric
from torch_geometric.data import DataLoader
from sklearn import metrics
folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'fold_6', 'fold_7', 'fold_8', 'fold_9']

class SimpleGCN(torch.nn.Module):
    def __init__(self, featurenum):
        super(SimpleGCN, self).__init__()
        self.conv1 = torchg.nn.GCNConv(featurenum, 32)
        self.conv2 = torchg.nn.GCNConv(32, 32)
        self.dropout = torch.nn.Dropout()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        # x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torchg.nn.global_mean_pool(x, data.batch)
        x = self.mlp(x)
        return torch.nn.functional.sigmoid(x)

class MPNNl(torchg.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNl, self).__init__(aggr='mean')  
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(in_channels, 4 * out_channels)
        # self.mlpf = torch.nn.Sequential(
        #     torch.nn.Linear(32, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 1)
        # )

    def forward(self, x, edge_index, edge_attr):
        # x, edge_index = data.x, data.edge_index
        x = self.lin(x)
        x = torch.reshape(x, (-1, 4, self.out_channels))
        rs = x[edge_index[0], edge_attr[:,0]]
        return self.propagate(edge_index, rs=rs, n= x.shape[0])

    def message(self, rs, n):
        return rs
        

class MPNN(torch.nn.Module):
    def __init__(self, featurenum):
        super(MPNN, self).__init__()
        self.conv1 = torchg.nn.GCNConv(featurenum, 32)
        self.conv2 = torchg.nn.GCNConv(32, 32)
        self.mpn = MPNNl(32, 32)
        self.dropout = torch.nn.Dropout()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.mpn(x, edge_index, data.edge_attr)
        x = torchg.nn.global_mean_pool(x, data.batch)
        x = self.dropout(x)
        x = self.mlp(x)
        return torch.nn.functional.sigmoid(x)
maxepoch = 40
if __name__ == '__main__':
    device = torch.device('cuda:1')
    lossf = torch.nn.BCELoss()
    ROCs = []
    PRCs = []
    for foldname in folds:
        trfname = 'pseudomonas/train_cv/' + foldname + '/train.csv'
        tefname = 'pseudomonas/train_cv/' + foldname + '/test.csv'
        dvfname = 'pseudomonas/train_cv/' + foldname + '/dev.csv'
        traindata = datap.snomaData(trfname, True)
        testdata = datap.snomaData(tefname, True)
        dataloader = DataLoader(traindata, batch_size= 32, shuffle = True)
        # simplegcn = SimpleGCN(40)
        simplegcn = MPNN(40)
        optim = torch.optim.Adam(simplegcn.parameters(), lr=1e-3)
        lossl = []
        losstestl = []
        accl = []
        aucl = []
        maxroc = 0
        for epo in range(maxepoch):
            print('epoch', epo)
            for idx, batch in enumerate(dataloader):
                out = simplegcn(batch)
                loss = lossf(out, batch.y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if idx %8 == 7: # eval
                    with torch.no_grad():
                        simplegcn.eval()
                        print('loss:', loss.item(), end = ' ')
                        tb = torchg.data.Batch.from_data_list(testdata)
                        out = simplegcn(tb)
                        losstest = lossf(out, tb.y).item()
                        print('losstest:', losstest, end = ' ')
                        lossl.append(loss) # lossl
                        losstestl.append(losstest) # losstestl
                        acc = ((out > 0.5).to(torch.float) == tb.y).sum().item() / len(testdata)
                        accl.append(acc) # accl
                        print('acc:', acc, end = ' ')
                        TPR = ((out > 0.5) & (tb.y == 1)).sum().item() / (tb.y == 1).sum().item()
                        TNR = ((out < 0.5) & (tb.y == 0)).sum().item() / (tb.y == 0).sum().item()
                        print('TPR, TNR:', TPR, TNR)
                        simplegcn.train()
                        rocauc = metrics.roc_auc_score(tb.y, out)
                        aucl.append(rocauc)
                        if (maxroc < rocauc):
                            maxroc = rocauc
                        p, r, thr = metrics.precision_recall_curve(tb.y, out)
                        prcauc = metrics.auc(r, p)
            
        # plott.plot(lossl, losstestl)
        plott.plot(accl, [float(i) for i in aucl])
        print('ROC AUC:', rocauc, 'PRC AUC', prcauc)
        ROCs.append(rocauc)
        PRCs.append(prcauc)
        # plott.plot(accl)
    print('final ROC AUC:', sum(ROCs) / len(ROCs), 'final PRC AUC:', sum(PRCs) / len(PRCs))