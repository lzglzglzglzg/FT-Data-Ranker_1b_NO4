import json
import numpy as np
from tqdm import tqdm

def get_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

data_path = ''
outfile_path = ''

data = get_jsonl(data_path)
n = len(data)


with open(outfile_path, 'w') as outfile:

    for i in tqdm(range(n)):
        if ('continue' in data[i]['instruction'] or 'Continue' in data[i]['instruction']) and len(data[i]['instruction']) < 10:
            print(data[i])
        else:
            json.dump(data[i], outfile)
            outfile.write('\n')
