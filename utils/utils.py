import os
import pandas as pd
import numpy as np
import random
import json
import scipy

from tqdm import tqdm


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    SOFT_GREEN = '\033[38;5;77m'
    SOFT_RED = '\033[38;5;124m'


def read_file(file_name, split_str=None, fraction=None):
    if 'jsonl' in file_name:
        datas = []
        with open(file_name, 'r', encoding='utf-8') as f:
            # Read part of file
            if fraction:
                total_lines = sum(1 for _ in f)
                f.seek(0)  # 回到文件开头
                read_lines = int(total_lines * fraction)
                for i, line in enumerate(f):
                    if i >= read_lines:
                        break
                    data = json.loads(line)
                    datas.append(data)
            else:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    datas.append(data)
        return datas
    elif 'json' in file_name:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    elif 'xlsx' in file_name:
        data = pd.read_excel(file_name)
        return data
    elif 'csv' in file_name:
        data = pd.read_csv(file_name)
        return data
    else:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = f.read()
        if split_str:
            elements = data.split(split_str)
            elements = [e.strip() for e in elements if e.strip()]
            return elements
        else:
            return data


def write_file(file_name, data, split_str=None):
    if type(data) is list:
        lists = data
        if 'jsonl' in file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                for element in lists:
                    json.dump(element, f, ensure_ascii=False, default=convert_np)
                    f.write('\n')
        else:
            split_str = '\n' if not split_str else split_str
            with open(file_name, 'w', encoding='utf-8') as f:
                for element in lists:
                    f.write(str(element))
                    f.write(split_str)
    elif type(data) is dict:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    elif isinstance(data, pd.DataFrame):
        if 'csv' in file_name:
            data.to_csv(file_name, index=False)
        elif 'xlsx' in file_name:
            data.to_excel(file_name, index=False)
    else:
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(str(data))


def merge_columns(old_data_path, new_data_path, columns=None, key='ID', save_path=None, output=False):
    old_data = read_file(old_data_path) if type(old_data_path) == str else old_data_path
    new_data = read_file(new_data_path) if type(new_data_path) == str else new_data_path
    columns = columns if columns else [c for c in new_data.columns if c not in old_data.columns]

    for index, row in old_data.iterrows():
        new_row = new_data[new_data[key] == row[key]]
        if new_row.shape[0] == 1:
            for column in columns:
                old_data.loc[index, column] = new_row.iloc[0][column]

    if save_path:
        write_file(save_path, old_data)
    if output:
        print("Update Columns %s!" % ('&'.join(columns)))
    return old_data


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M, base=2) + 0.5 * scipy.stats.entropy(q, M, base=2)
