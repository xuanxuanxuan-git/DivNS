from .metrics import *
from .parser import parse_args

import random
import torch
import math
import numpy as np
import multiprocessing
import heapq
from time import time
import csv
import os
import functools

# cores = multiprocessing.cpu_count() // 2
cores=2
args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
# device = torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


# evaluation protocol: evaluate against all items not in the training set
def test_one_user(x, train_user_set, test_user_set):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []

    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def save_ind_results(test_user_ids, ind_results, train_user_set):
    assert len(test_user_ids) == len(ind_results)
    
    user_num_inter = {user_id: len(interactions) for user_id, interactions in train_user_set.items()}
    folder_path = "results_10/{}/{}".format(args.dataset, args.rec_model)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # file_path = folder_path + "/{}_{}_{}_{}.csv".format(args.dataset, args.ns, args.n_negs, args.K)
    file_path = folder_path + "/{}_{}_{}_{}_{}.csv".format(args.dataset, args.ns, args.n_negs, args.K, args.num_group)
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'num_int', 'precision', 'recall', 'ndcg', 'hit'])
        
        for user_id, user_metrics in zip(test_user_ids, ind_results):
            try:
                num_int = user_num_inter[user_id]
            except:
                num_int = 0
            writer.writerow([user_id] + [num_int] + user_metrics)

    print("test result saved successfully.")
 

    
def test(model, user_dict, n_params, mode='test', save=False):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}
    individual_record = []
    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    # global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    train_user_group = user_dict["train_user_group"]
    if mode == 'test':
        test_user_set = user_dict['test_user_set']
    elif mode == "valid":
        test_user_set = user_dict['valid_user_set']
        if test_user_set is None:
            print("No validation set.")
            # test_user_set = user_dict['test_user_set']
    # pool = multiprocessing.Pool(cores)
    # print("len test user set:", len(test_user_set))
    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batches = n_test_users // u_batch_size + 1

    count = 0

    user_gcn_emb, item_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batches):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start:end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batches = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batches):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embeddings = item_gcn_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embeddings).detach().cpu()

                rate_batch[:, i_start:i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embeddings = item_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embeddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)

        # batch_result = pool.map(test_one_user, user_batch_rating_uid)
        
        map_result = map(functools.partial(test_one_user,train_user_set=train_user_set, test_user_set=test_user_set), list(user_batch_rating_uid)) #, train_user_set, test_user_set)
        # print("len of batch result:", len(batch_result))
        batch_result = list(map_result)

        count += len(batch_result)
        # print("count", count)
        # # count += len(list(user_batch_rating_uid))
        # print("length", len(list(batch_result)))

        # for re in batch_result:
        #     print("result", re)
        #     result['precision'] += re['precision']/n_test_users
        #     print(re["precision"]/n_test_users)
        #     result['recall'] += re['recall']/n_test_users
        #     result['ndcg'] += re['ndcg']/n_test_users
        #     result['hit_ratio'] += re['hit_ratio']/n_test_users
        #     result['auc'] += re['auc']/n_test_users
        
        for re in batch_result:
            result['precision'] += re['precision'][0]
            result['recall'] += re['recall'][0]
            result['ndcg'] += re['ndcg'][0]
            result['hit_ratio'] += re['hit_ratio'][0]
            result['auc'] += re['auc']
            # if mode == "test":
            individual_record.append([re["precision"][0], re["recall"][0], re["ndcg"][0], re["hit_ratio"][0]])

    # if calculate ndcg disp, put 2 in; if calculate recall, put 1 in.
    result["recall-disp"], result["group-recall"] = recall_disp_simple(test_users, train_user_group, [record[1] for record in individual_record])
    result['precision'] = result['precision']/n_test_users
    result['recall'] = result['recall']/n_test_users
    result['ndcg'] = result['ndcg']/n_test_users
    result['hit_ratio'] = result['hit_ratio']/n_test_users
    result['auc'] = result['auc']/n_test_users

    if mode == "test" and save: 
        save_ind_results(test_users, individual_record, train_user_set)
        # result["recall-disp"] = recall_disp
    
    assert count == n_test_users
    # pool.close()
    del rate_batch, item_batch, user_dict, model, batch_result, re
    return result
