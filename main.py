import os
import random
import sys
import torch
import numpy as np
import copy
from time import time
from copy import deepcopy
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping, save_best_recall_group, save_item_selections, run_best_model_group
from utils.evaluate_sample import *
from collections import Counter, defaultdict
from torch.nn.utils.rnn import pad_sequence
import wandb
wandb.disabled = True

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, train_user_group, start, end, item_ids, sample_pb, n_negs=1, k_group=1):
    # for each batch, randomly sample n_negs (default=6)*K negative items, 
    # those subsampled data will be further down-sampled.
    # out of n_negs*K items, K items will be further selected during the model training.
    def sampling(user_item, train_set, user_group, n_negs, item_ids, sample_pb):
        n = n_negs * K
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            n = k_group[int(user_group[user])]*n_negs
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items)) if args.ns != "pns" else np.random.choice(item_ids, size=1, p=sample_pb)[0]
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    ratio_mapping = {group_id:ratio for group_id, ratio in zip(range(args.num_group), k_group)}

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]    
    
    group_ids = [train_user_group[user_id] for user_id in feed_dict["users"].tolist()]
    feed_dict["neg_ratio"] = torch.tensor([ratio_mapping[group_id] for group_id in group_ids]).to(device)

    sampled_neg_items = sampling(entity_pairs, train_pos_set, train_user_group,
                                                       n_negs, item_ids, sample_pb)
    neg_items_tensors = [torch.tensor(neg_items) for neg_items in sampled_neg_items]
    feed_dict["neg_items"] = pad_sequence(neg_items_tensors, batch_first=True, padding_value=-1).to(device)
    feed_dict["user_group"] = torch.tensor(group_ids).to(device)

    # feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs, 
    #                                                    train_pos_set, train_user_group, n_negs, item_ids, sample_pb)).to(device)

    # feed_dict["neg_items"].shape: [batch_size, n_negs*max(k_group)]

    return feed_dict


def calc_frequency(all_data):   # Calculate item frequency and sample probability
    # its probability to be sampled is the its own popularity divided by global popularity
    items = all_data[:, 1]
    item_counts = Counter(items)
    item_ids = list(item_counts.keys())
    item_freq = list(item_counts.values())
    global_pb = np.power(item_freq, 0.75)
    sample_pb = global_pb / np.sum(global_pb)
    return sample_pb, item_ids

def run_on_test(model, user_dict, n_params, save=False):
    # use the saved best model to run on test set
    test_ret = test(model, user_dict, n_params, mode='test', save=save)
    return test_ret

def update_ratio(avg_loss, group_loss, prev_group_k):
    new_group_k = [0]*len(prev_group_k)
    avg_loss = avg_loss/len(prev_group_k)
    for index, loss in enumerate(group_loss):
        if loss < avg_loss.item():
            new_group_k[index] = min(16, prev_group_k[index] + 1)
        elif loss > avg_loss.item():
            new_group_k[index] = max(1, prev_group_k[index] - 1)
        else:
            new_group_k[index] = prev_group_k[index]
    return new_group_k
        


def start_wandb(args):
    logger = wandb.init(
        project="diverse_ns", 
        config = args,
        # name = "beauty_test_1",
    )
    auto_run_name = wandb.run.name
    # auto_run_num = auto_run_name.rsplit('-', 1)[-1]

    # wandb.run.name = args.dataset + "-" + args.ns + "-" + str(args.K) + "-" + "10" + "-" + auto_run_num
    # wandb.run.save()
    return logger


def define_model(args, n_params, norm_mat, device):
    if args.rec_model == 'lightgcn':
        from modules.LightGCN import LightGCN
        model = LightGCN(n_params, args, norm_mat).to(device)
    elif args.rec_model == "ngcf":
        from modules.NGCF import NGCF
        model = NGCF(n_params, args, norm_mat).to(device)
    elif args.rec_model == "mf":
        from modules.MF import MF
        model = MF(n_params, args).to(device)
    elif args.rec_model == "ncf":
        from modules.NCF import NCF
        model = NCF(n_params, args).to(device)
    elif args.rec_model == "mlp":
        from modules.NCF import MLP 
        model = MLP(n_params, args).to(device)
    elif args.rec_model == "gmf":
        from modules.NCF import GMF
        model = GMF(n_params, args).to(device)

    return model

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    """Initialise wandb logger"""
    logger = start_wandb(args)
    print(args)

    """build dataset"""
    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    print("training set size:", train_cf_size)
    if args.ns == "pns":
        item_sample_prob, item_ids = calc_frequency(train_cf)
    else:
        item_sample_prob, item_ids = 0, 0
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K
    k_group = [K]*args.num_group

    """define model"""
    model = define_model(args, n_params, norm_mat, device)


    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    cur_best_group_rec = [0]*args.num_group
    best_group_epoch = [0]*args.num_group
    cur_best_model_group = [0]*args.num_group
    stopping_step = 0
    should_stop = False
    all_pb = []

    print("start training ...")
    for epoch in range(args.epoch):

        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)     # size: number of interactions in training set * 2
        epoch_user_group = [user_dict["train_user_group"][uid] for uid in (train_cf_[:,0]).tolist()]

        """training"""
        model.train()
        loss, s = 0, 0
        epoch_avg_loss = 0
        epoch_group_loss = torch.zeros(args.num_group) 

        """distribute recycled negatives"""

        train_s_t = time()
        while s < len(train_cf):
            current_batch_size = min(args.batch_size, len(train_cf) - s)
            batch = get_feed_dict(train_cf_,
                                user_dict['train_user_set'], user_dict["train_user_group"],
                                s, s + current_batch_size,
                                item_ids, 
                                item_sample_prob, n_negs, k_group)
            
            batch_loss, batch_pos_scores, batch_neg_scores = model(epoch, batch)      # this is where forward is called
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += current_batch_size

        """compute the cache"""
        model.recycle_negs_idx = defaultdict(list)
        model.recycle_cached_samples()      # recycle is fast and takes the same time each epoch (0.9s)
        model.cached_negs_idx = defaultdict(list)
        model.latest_used_negs_idx = defaultdict(list)
        
        train_e_t = time()
        logger.log({"epoch": epoch, "train_loss":loss.item(), "train_time":train_e_t-train_s_t})
        logger.log({"k_group": k_group})
        

        if epoch % 5 == 0 and epoch != 0:
            """evaluation"""
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "eval time(s)", "Loss", "recall", "ndcg", "precision", "recall-disp", "group-recall"]

            model.eval()
            ## ------ run on test set ----------
            test_s_t = time()
            test_ret = run_on_test(model, user_dict, n_params, save=False)
            test_e_t = time()
            train_res.add_row(
                [epoch, round(train_e_t - train_s_t, 4), round(test_e_t - test_s_t, 4), loss.item(), test_ret['recall'], test_ret['ndcg'],
                 test_ret['precision'], test_ret['recall-disp'], [round(value, 4) for value in test_ret["group-recall"]]])
            
            ## -------- run on validation set -------
            if user_dict['valid_user_set'] is None:
                print("No validation set")
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, round(train_e_t - train_s_t, 4), round(test_e_t - test_s_t, 4), loss.item(), valid_ret['recall'], valid_ret['ndcg'],
                     valid_ret['precision'], valid_ret['recall-disp'],
                      [round(value, 4) for value in valid_ret["group-recall"]]])
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 20 successive steps.
            # if validation set exists, then early stopping is based on the validation set.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc',
                                                                    flag_step=20)
            
            # save the best parameters for each group
            cur_best_group_rec, cur_best_model_group, best_group_epoch = save_best_recall_group(cur_best_group_rec, cur_best_model_group, best_group_epoch, valid_ret, model, epoch)
            print("Current valid group recall: {}, \nBest valid group recall so far: {}".format(valid_ret["group-recall"], cur_best_group_rec))

            logger.log({"eval_epoch": epoch, 
                        "eval_time": test_e_t - test_s_t,
                        "eval_ndcg": valid_ret["ndcg"], 
                        "eval_recall": valid_ret["recall"],
                        })
            if should_stop:
                break

            if valid_ret['recall'][0] == cur_best_pre_0:
                # best_model = model       # save the model
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                logger.log({"early_stop_epoch": epoch})
                
                """save weight"""
                if args.save:
                    torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')
        else:
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, best recall@20 on the valid set: %.4f at epoch %d' % (epoch, cur_best_pre_0, best_epoch))
    model.load_state_dict(best_model_wts)
    model.eval()
    best_test_result = run_on_test(model, user_dict, n_params, save=True)
    
    print("Final best results on test set -- ndcg: %.4f, recall: %.4f, recall-disp: %.4f" % (best_test_result["ndcg"], best_test_result["recall"], best_test_result["recall-disp"]))
    logger.log({"test_ndcg": best_test_result["ndcg"], "test_recall": best_test_result["recall"], "test_recall_disp": best_test_result["recall-disp"]})

    ## only output the training item selections up to the best epoch
    ## epoch starts from 0, so the counter needs to add by 1.
    # used_item_selections = model.item_selections[:best_epoch+1]
    # save_item_selections(used_item_selections, args)

