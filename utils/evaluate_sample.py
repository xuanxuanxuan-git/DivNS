import numpy as np
import torch
import pickle
import os
# all_training_scores[epoch] = [[pos_scores],[neg_scores]]
# pos_scores: [pos_instances]
# neg_scores: [pos_instances, K]

def compute_pb(pos_scores, neg_scores):
    pos_scores = torch.sigmoid(pos_scores)
    neg_scores = torch.sigmoid(neg_scores)
    pos_scores = pos_scores.unsqueeze(1)
    prob = pos_scores * neg_scores
    prob = prob.mean(dim=1)
    return prob

def compute_good_count(p_a, p_b):
    good_count = []
    for group in p_a:
        group_good_count = p_a[group]*(1-p_b[group])
        good_count.append(group_good_count)
    return good_count

def compute_bad_count(p_a, p_b):
    bad_count = []
    for group in p_a:
        group_bad_count = p_b[group]*(1-p_a[group])
        bad_count.append(group_bad_count)
    return bad_count

def compute_epoch_reliability(epoch_pos_scores, epoch_neg_scores, user_group):
    group_pb = {}
    p_b = compute_pb(epoch_pos_scores, epoch_neg_scores)  # size: [pos_instances]
    
    for group_id in range(max(user_group)+1):
        group_mask = (torch.tensor(user_group) == group_id) 
        group_scores = p_b[group_mask]
        average_score = group_scores.mean() 
        group_pb[group_id] = average_score.item()
    return group_pb

def save_effectiveness_results(group_good, group_bad, args):
    result_dict = {
        "ns":args.ns,
        "rec_model": args.rec_model,
        "ratio": args.K,
        "group_good": group_good,
        "group_bad": group_bad,
    }
    
    file_path = f"results/train_effect/{args.dataset}_{args.rec_model}.pkl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "ab") as output_file:
        pickle.dump(result_dict, output_file)
    print("Training effectiveness saved successfully.")


def compute_effectiveness(all_reliability, best_epoch, args):
    p_a = all_reliability[best_epoch]  #{0: , 1: , 2: , 3: }
    all_good_count = []
    all_bad_count = []
    for e in range(best_epoch):
        current_p_b = all_reliability[e]
        current_good_count = compute_good_count(p_a, current_p_b) # [ , , , ,]
        current_bad_count = compute_bad_count(p_a, current_p_b)

        all_good_count.append(current_good_count)
        all_bad_count.append(current_bad_count)

    # mean for each user group
    avg_good_count = np.mean(np.array(all_good_count), axis=0)
    avg_bad_count = np.mean(np.array(all_bad_count), axis=0)
    save_effectiveness_results(avg_good_count, avg_bad_count, args)
    return avg_good_count, avg_bad_count
    