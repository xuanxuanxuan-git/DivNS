import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score


def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    # if all_pos_num == 0:
    #     return 0
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def recall_disp_simple(test_users, user_group_dict, recalls):
    group_recall_sum = defaultdict()
    group_user_count = defaultdict()  

    # Aggregate recalls by group
    for user, recall in zip(test_users, recalls):
        group = user_group_dict[user]
        if group not in group_recall_sum:
            group_recall_sum[group] = 0.0
            group_user_count[group] = 0
        group_recall_sum[group] += recall
        group_user_count[group] += 1

    # Calculate average recall per group
    group_recall = {group: group_recall_sum[group] / group_user_count[group] for group in group_recall_sum}
    group_recall = {k:v for k,v in sorted(group_recall.items())}
    # print("sorted group recall:", group_recall)
    
    mean_recall = np.mean(recalls)
    std_recall = np.std(list(group_recall.values()))
    recall_disp = std_recall/mean_recall
    return recall_disp, list(group_recall.values())

# if __name__ == "__main__":
#     dataset_name = "pinterest"
#     cf_model = "mf"
#     neg_sampler = "dns"

#     data_file_path = "../results/{}/{}/{}_{}_6_1.csv".format(dataset_name, 
#                                                              cf_model, dataset_name, 
#                                                              neg_sampler)

#     data_df = pd.read_csv(data_file_path, header=0)
#     recall_disp = recall_disparity(data_df)
#     print(f"recall disparity for {dataset_name} with {neg_sampler} and {cf_model} is: {recall_disp}")
