import argparse
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

path = '../data/data.jsonl'
model_path = '/sshfs/pretrains/BAAI/'

key_with_output = ['instruction', 'input', 'output']
key_without_output = ['instruction', 'input']
data = []
embeds_with_output = []
embeds_without_output = []

def get_text(x, key):
    text = ''
    for i in key:
        text += x[i]
        text += '\n'
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='bge-base-en-v1.5')
    args = parser.parse_args()

    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # data = data[:100]

    print(f"Processing {len(data)} examples")

    model = SentenceTransformer(model_path + args.model)

    for i in tqdm(range(len(data))):
        text_with_output = get_text(data[i], key_with_output)
        text_without_output = get_text(data[i], key_without_output)
        embed_with_output = model.encode(text_with_output)
        embed_without_output = model.encode(text_without_output)
        embeds_with_output.append(np.array(embed_with_output))
        embeds_without_output.append(np.array(embed_without_output))
        # if i == 0:
        #     print('第一个样例')
        #     print('text_with_output:' + text_with_output)
        #     print('text_without_output:' + text_without_output)
        #     print('embed_with_output.shape:', embed_with_output.shape)
        #     print('embed_without_output.shape:', embed_without_output.shape)
        #     print('embed_with_output:', embeds_with_output)
        #     print('embed_without_output:', embeds_without_output)

    embeds_with_output = np.vstack(embeds_with_output)
    embeds_without_output = np.vstack(embeds_without_output)

    np.save(f"embeds_with_output.npy", embeds_with_output)
    np.save(f"embeds_without_output.npy", embeds_without_output)
