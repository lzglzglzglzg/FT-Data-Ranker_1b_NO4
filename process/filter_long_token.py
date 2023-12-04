import json
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

model_path = ''
data_path = ''
out_path = ''
length_path = 'length.npy'
new_length_path = 'new_length.npy'

TOKEN_MAX_LENGTH = 1024 # 要求小于1024，context长度为1024，包含1个特殊token
PROMPT_DICT = {
    "en": {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )},
    "zh": {
    "prompt_input": (
        "以下是描述任务的指示，配有提供进一步上下文的输入，编写一个适当的回应完成请求\n\n"
        "### 指示：\n{instruction}\n\n### 输入：\n{input}\n\n### 回应："
    ),
    "prompt_no_input": (
        "以下是描述任务的指示，编写一个适当的回应完成请求\n\n"
        "### 指示：\n{instruction}\n\n### 回应："
    )}
}

enc = AutoTokenizer.from_pretrained(model_path)
dataset = []
outdata = []
prompt_input = PROMPT_DICT['en']['prompt_input']
prompt_no_input = PROMPT_DICT['en']['prompt_no_input']

with open(data_path, 'r', encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

for data in dataset:
    if "instruction" in data and "output" in data:
        data["target"] = data["output"]
        if data.get("input", "") != "":
            data["source"] = prompt_input.format_map(data)
        else:
            data["source"] = prompt_no_input.format_map(data)
    else:
        raise RuntimeError(f"{data}")

length = []
new_length = []
for i in tqdm(range(len(dataset))):
    l = len(enc.tokenize(dataset[i]['source'])) + len(enc.tokenize(dataset[i]['target']))
    length.append(l)
    if l < TOKEN_MAX_LENGTH:
        outdata.append({k : v for k, v in dataset[i].items() if k in ['instruction', 'input', 'output']})
        new_length.append(l)

i = 0
with open(out_path, 'w') as outfile:
    for line in outdata:
        if i != 0:
            outfile.write('\n')
        json.dump(line, outfile)
        i += 1
    print('end')

# np.save(length_path, np.array(length))
# np.save(new_length_path, np.array(new_length))