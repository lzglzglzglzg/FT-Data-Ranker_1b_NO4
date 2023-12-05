# FT-Data-Ranker

## 各文件夹内容
- data:raw_data.jsonl处理后的数据
- finetuned_model_1b:微调后的1B模型
- lm-training:官方训练代码
- process:数据处理代码，细节见 [process/README.md](https://github.com/lzglzglzglzg/FT-Data-Ranker_1b_NO4/blob/master/process/README.md)

## 方案

- dataset quantization:通过相似度匹配采样数据，以保证数据的多样性
  - 论文链接：https://arxiv.org/abs/2308.10524
  - GitHub：https://github.com/magic-research/Dataset_Quantization
- data-juicer:比赛官方的数据处理包，包含各种算子
  - GitHub:https://github.com/alibaba/data-juicer

关于比赛数据的处理细节见 [process/README.md](https://github.com/lzglzglzglzg/FT-Data-Ranker_1b_NO4/blob/master/process/README.md)