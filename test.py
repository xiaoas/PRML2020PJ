import torch
import torch_geometric
import csv
import pysmiles
cnt = {}
def readf(fname):
    with open(fname) as csvf:
        reader = csv.reader(csvf)

        a = 0
        for row in reader:
            if a == 0:
                a = 1
                continue
            gf = pysmiles.read_smiles(row[0])
            for ele in gf.nodes(data='element'):
                ele = ele[1]
                # if ele in cnt:
                #     cnt[ele] += 1
                # else:
                #     cnt[ele] = 1
                if ele not in cnt:
                    cnt[ele] = len(cnt)
        
readf('../pseudomonas/train_cv/fold_0/train.csv')
readf('../pseudomonas/train_cv/fold_0/test.csv')
print(cnt)