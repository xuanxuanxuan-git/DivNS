import os
import re
import pickle
import copy
from .evaluate import test
import pandas as pd

def save_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is triggered for {} steps, recorded value: {}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def update_best_res(log_value, best_value, expected_order='acc'):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        best_value = log_value

    return best_value


def save_best_recall_group(cur_best_group_rec, cur_best_model, best_group_epoch, new_valid_res, new_model, cur_epoch):
    """
    save the best performing model for each user group.
    """
    new_group_recall = new_valid_res["group-recall"]
    for i in range(len(cur_best_group_rec)):
        if new_group_recall[i]>cur_best_group_rec[i]:
            cur_best_group_rec[i] = new_group_recall[i]
            cur_best_model[i] = copy.deepcopy(new_model.state_dict())
            best_group_epoch[i] = cur_epoch

    return cur_best_group_rec, cur_best_model, best_group_epoch


def run_best_model_group(model, best_model_group, test_users, n_params):
    best_group_recall = [0]*len(best_model_group)
    for group_id, group_model_wts in enumerate(best_model_group):
        model.load_state_dict(group_model_wts)
        model.eval()
        best_group_test_result = test(model, test_users, n_params, mode="test")
        best_group_recall[group_id] = best_group_test_result["group-recall"][group_id]
    return best_group_recall


def save_item_selections(used_item_selections, args):
    folder_path = "selections/{}/{}".format(args.dataset, args.rec_model)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path =  f"{folder_path}/{args.dataset}_{args.ns}_{args.K}.csv"
    
    rows = []
    for i in range(used_item_selections[0].shape[0]):  # Iterate over rows
        row = []
        for tensor in used_item_selections:
            values = tensor[i].tolist()
            values_str = ",".join(map(str, values))
            row.append(values_str) 
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=[f'epoch_{i}' for i in range(len(used_item_selections))])

    df.to_csv(file_path, index=False)
    print("Item selection results for training saved successfully.")


