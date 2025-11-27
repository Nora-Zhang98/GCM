import os
from maskrcnn_benchmark.modeling.utils import cat
import torch
from tqdm import tqdm

n_class, feat_dim = 30, 4096 # vg: 50, gqa:100, oiv6: 30
path = './output/relation_baseline/infer_train_feat'
save_path = './output/relation_baseline'
files = next(os.walk(path))[2]
rep_container = [[] for _ in range(n_class)]

def rep_rec(rep, label):
    for i in range(len(label)):
        idx = int(label[i].item())
        rep_container[idx-1].append(rep[i])

def mean_var_item(input_list):
    container = torch.stack(input_list)
    if torch.cuda.is_available():
        container = container.cuda()
    mean = torch.mean(container, dim=0, keepdim=True)
    var = torch.var(container, dim=0, keepdim=True)
    # print(mean)
    # print(var)
    del container
    return mean, var

def mean_var(feat_dim):
    all_mean, all_var = [], []
    print("fg predicate mean/var calculate.....")
    for i in tqdm(range(len(rep_container))):
        if len(rep_container[i]) == 0:
            all_mean.append(torch.zeros((1, feat_dim)))
            all_var.append(torch.zeros((1, feat_dim)))
        else:
            mean, var = mean_var_item(rep_container[i])  # calculate mean and var
            all_mean.append(mean.cpu())
            all_var.append(var.cpu())

    all_mean = cat(all_mean, dim=0)
    all_var = cat(all_var, dim=0)
    result = {
        "all_mean": all_mean,
        "all_var": all_var,
    }
    torch.save(result, os.path.join(save_path, "all_mean_var_oiv6_vtranse.pkl"))

for name in tqdm(files):
    fp = os.path.join(path, name)
    fp_data = torch.load(fp, map_location=torch.device('cpu'))
    fg_lb, rep = fp_data['fg_label'], fp_data['rel_rep_norm']
    rep_rec(rep, fg_lb)
    del fp_data

mean_var(feat_dim)

