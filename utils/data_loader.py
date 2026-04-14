import numpy as np
import scipy.sparse as sp
import math
import pandas as pd

from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)
train_user_group = dict()

train_item_set = defaultdict(list)
test_item_set = defaultdict(list)
valid_item_set = defaultdict(list)

train_item_pop = defaultdict(int)
test_item_pop = defaultdict(int)
valid_item_pop = defaultdict(int)

dataset_list = ['ml-1m', 'beauty', 'book-crossing', 'pinterest',
                 'baby', 'home', 'toy', 'lastfm', 'yelp2022']


def read_cf_amazon(file_name):
    # return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]
    dataset_df = pd.read_csv(file_name, header=0)
    return dataset_df.values

def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def assign_user_group(train_user_set, num_group):
    # for each user, calculate number of interacted items
    activities = {key: len(value) for key, value in train_user_set.items()}
    train_user_df = pd.DataFrame(list(activities.items()), columns=["user_id", "num_int"])
    train_user_df["quantile"] = pd.qcut(train_user_df["num_int"].rank(method="first"), num_group, labels=False)
    # train_user_group = dict(zip(train_user_df["user_id"], train_user_df["quantile"]))
    for user_id, num_int in zip(train_user_df["user_id"], train_user_df["quantile"]):
        train_user_group[user_id] = num_int
    # return train_user_group


def statistics(train_data, valid_data, test_data):
    global n_users, n_items, n_inter
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1
    n_inter = train_data.shape[0] + test_data.shape[0] + valid_data.shape[0]

    if dataset not in dataset_list:
        n_items -= n_users
        # remap [n_users, n_users+n_items] to [0, n_items]
        train_data[:, 1] -= n_users
        valid_data[:, 1] -= n_users
        test_data[:, 1] -= n_users

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))
    
    assign_user_group(train_user_set, args.num_group)

    for u_id, items in train_user_set.items():
        for i_id in items:
            train_item_set[i_id].append(u_id)
    for u_id, items in test_user_set.items():
        for i_id in items:
            test_item_set[i_id].append(u_id)
    for u_id, items in valid_user_set.items():
        for i_id in items:
            valid_item_set[i_id].append(u_id)

    for i_id, users in train_item_set.items():
        train_item_pop[i_id] = len(users)
    for i_id, users in test_item_set.items():
        test_item_pop[i_id] = len(users)
    for i_id, users in valid_item_set.items():
        valid_item_pop[i_id] = len(users)

def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat)


def load_data(model_args):
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    if dataset not in dataset_list:
        read_cf = read_cf_yelp2018
    else:
        read_cf = read_cf_amazon
    
    print("loading dataset:", args.dataset)
    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.csv')
    test_cf = read_cf(directory + 'test.csv')
    if dataset in dataset_list:
        valid_cf = read_cf(directory + 'valid.csv')
    else:
        valid_cf = test_cf
    statistics(train_cf, valid_cf, test_cf)

    print('building the adj mat ...')
    norm_mat = build_sparse_graph(train_cf)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_inter': int(n_inter)
    }
    print(n_params)
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set if args.dataset in dataset_list else None,
        'test_user_set': test_user_set,
        # add item set
        'train_item_set': train_item_set,
        # assign user group
        "train_user_group": train_user_group
    }
    # item_dict = {
    #     'train_item_set': train_item_set,
    #     'valid_item_set': valid_item_set if args.dataset in dataset_list else None,
    #     'test_item_set': test_item_set
    # }
    # item_pop_dict = {
    #     'train_pop_set': train_item_pop,
    #     'valid_pop_set': valid_item_pop if args.dataset in dataset_list else None,
    #     'test_pop_set': test_item_pop
    # }
    # for NNCF
    # pop_weight = []
    # for i_id in range(int(n_items)):
    #     pop_weight.append(math.pow(train_item_pop[i_id], 0.75))

    print('loading over ...')
    # return train_cf, user_dict, n_params, norm_mat, pop_weight
    return train_cf, user_dict, n_params, norm_mat

