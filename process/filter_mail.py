import json
import numpy as np
from tqdm import tqdm

def get_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

data_path_1 = ''
data_path_2 = ''
outfile_path = ''

data_1 = get_jsonl(data_path_1)
data_2 = get_jsonl(data_path_2)
n = len(data_1)

num_same = 0
num_not_same = 0

with open(outfile_path, 'w') as outfile:

    for i in tqdm(range(n)):
        if data_1[i] != data_2[i]:
            num_not_same += 1
            print(data_1[i])
            print(data_2[i])
        else:
            num_same += 1
            json.dump(data_1[i], outfile)
            outfile.write('\n')


print(num_same)
print(num_not_same)