import argparse
from tqdm import *
import numpy as np
import torch
import json
import math
import util.utils as utils
from dq.methods.methods_utils.submodular_function import GraphCut
from dq.methods.methods_utils.submodular_optimizer import NaiveGreedy
import sys
import os
import time

# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

data_path = '../data/data.jsonl'
embeding_path = 'embeds_with_output.npy'
outfile_path = 'data_dq.jsonl'
filter_outfile_path = 'data_dq_filter.jsonl'
filter_embeds_outfile_path = 'embeds_with_output_filter.npy'

def random_sample(data, n=1000):
    indices = np.random.choice(len(data), n, replace=False)
    return [data[i] for i in indices]

def dataset_quantization(data, embeddings_data, ratio, k):
    n = int(len(data) * ratio)
    bins_n = int(n / k)
    budget_n = len(data) // k

    embeddings_original = embeddings_data
    embeddings = embeddings_data

    indices_original = np.arange(len(data))
    indices = indices_original.copy()

    print(f"total: {len(data)}, ratio:{ratio}, n: {n}, k: {k}, budget_n: {budget_n}, bins_n: {bins_n}, embeddings: {len(embeddings)}")

    # sim_matrix = lambda a, b: embeddings[a] @ embeddings[b].T

    def sim_matrix(a, b):
        a_tensor = torch.tensor(embeddings[a],device="cuda:3")
        b_tensor = torch.tensor(embeddings[b],device="cuda:3")
        m = a_tensor@b_tensor.T
        return m.cpu().numpy()

    # bin generation
    bins = []
    for i in range(k):
        print(f"bin {i}/{k}")
        print('embeddings长度', len(embeddings))
        submod_f = GraphCut(index=indices, similarity_kernel=sim_matrix)
        submod_opt = NaiveGreedy(args=None, index=indices, budget=budget_n)
        result_indices = submod_opt.select(
            gain_function=submod_f.calc_gain,
            update_state=submod_f.update_state,
        )

        bins.append(result_indices)
        indices = np.delete(indices_original, np.concatenate(bins))
        embeddings = np.delete(embeddings_original, np.concatenate(bins), axis=0)

    # bin sampling
    index = []
    assert len(bins) == k
    for i in range(k):
        sampled_indices = random_sample(bins[i], n=bins_n)
        index.extend(sampled_indices)
    data = [data[i] for i in index]
    embeddings = [embeddings_data[i] for i in index]
    print(f"sampled: {len(data)} examples")

    return [data, embeddings]

if __name__ == "__main__":
    # 自定义目录存放日志文件
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--ratio_first", type=float, default=0.015)
    parser.add_argument("--ratio_second", type=float, default=0.5)
    parser.add_argument("--group_num", type=int, default=400)
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    data = []
    print('open ' + data_path)
    with open(data_path, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    embeddings = np.load(embeding_path)

    print(f'data_len:{len(data)},embeddings_len:{len(embeddings)}')
    
    # 设置test
    # data = data[:10000]
    # embeddings = embeddings[:10000]

    k = args.k
    group_num = args.group_num
    group_size = int(len(data) // group_num)
    ratio_first = args.ratio_first
    ratio_second = args.ratio_second
    data = data[:(group_num * group_size)]
    embeddings = embeddings[:(group_num * group_size)] 
    print(f"k: {k}, group_num: {group_num}, group_size: {group_size}, ratio_first: {ratio_first}, ratio_second: {ratio_second}, data_len: {len(data)}, embeddings: {len(embeddings)}")

    data_filter = []
    embeddings_filter = []

    if ratio_first != 1:
        for i in range(group_num):
            print(f'过滤第{i}组')
            filter_result = dataset_quantization(data[i * group_size:(i + 1) * group_size], embeddings[i * group_size:(i + 1) * group_size], ratio=ratio_first, k=k)
            data_filter.extend(filter_result[0])
            embeddings_filter.extend(filter_result[1])
        embeddings_filter = np.array(embeddings_filter)

        pbar = tqdm(total=len(data_filter))
        with open(filter_outfile_path, 'w') as outfile:
            for i in range(len(data_filter)):
                if i > 0:
                    outfile.write('\n')
                json.dump(data_filter[i], outfile)
                pbar.update(1)
        np.save(filter_embeds_outfile_path, embeddings_filter)
    else:
        data_filter = data
        embeddings_filter = embeddings
    

    print(f'data_filter_num:{len(data_filter)}, embeddings_filter:{embeddings_filter}')
    data = dataset_quantization(data_filter, embeddings_filter, ratio=ratio_second, k=k)[0]
    print(f"DQ: {len(data)} examples")

    pbar = tqdm(total=len(data))
    with open(outfile_path, 'w') as outfile:
        for i in range(len(data)):
            if i > 0:
                outfile.write('\n')
            json.dump(data[i], outfile)
            pbar.update(1)
    pbar.close()
    
    print('sample end')