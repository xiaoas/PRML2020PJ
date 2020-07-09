import torch
import torch_geometric
import csv
import pysmiles
elemap = {'C': 0, 'O': 1, 'N': 2, 'P': 3, 'S': 4, 'Cl': 5, 'Na': 6, 'F': 7, 'Br': 8, 'Sb': 9, 'K': 10, 'Ca': 11, 'Gd': 12, 'I': 13, 'Li': 14, 'Bi': 15, 'As': 16, 'Hg': 17, 'Zn': 18, 'Si': 19, 'Pb': 20, 'Fe': 21, 'Pt': 22, 'Se': 23, 'Co': 24}
def snomaData(fname, popposi = False):
    datas = []
    with open(fname) as csvf:
        reader = csv.reader(csvf)
        a = 0
        for row in reader:
            if a == 0:
                a = 1
                continue
            gf = pysmiles.read_smiles(row[0]) # 'CCCCCc1c2c(cc(O)c1C(=O)O)OC(=O)c1c(cc(OC)cc1C(=O)CCCC)O2'
            x = torch.empty((gf.number_of_nodes(), 40))
            edge_index = torch.empty((2, 2 * gf.number_of_edges()))
            edge_attr = torch.empty((2 * gf.number_of_edges(), 1))
            for idx in range(gf.number_of_nodes()):
                x[idx] = torch.cat((torch.eye(32)[elemap[gf.nodes(data='element')[idx]]], torch.eye(8)[gf.nodes(data='hcount')[idx]]))
            for idx, edge in enumerate(gf.edges):
                edge_index[:, idx * 2] = torch.tensor(edge)
                edge_index[:, idx * 2 + 1] = torch.tensor(edge[::-1])
            for idx, edge in enumerate(gf.edges(data='order')):
                edge_attr[idx * 2, 0] = edge[2]
                edge_attr[idx * 2 + 1, 0] = edge[2]
            datum = torch_geometric.data.Data(x= x, edge_index=edge_index.to(torch.long), edge_attr=edge_attr, y = torch.tensor([[int(row[1])]], dtype = torch.float))
            if datum.y == 1 and popposi:
                datas += 24 * [datum]
            else:
                datas.append(datum)
    return datas