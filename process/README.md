# 执行过程

- 1 filter_long_token.py 过滤超过1024 token 的样本
- 2 embed.py 计算每个样本的语义向量
- 3 sample.py 根据样本的语义相似度进行采样
- 4 使用data_juicer 根据 config_mapper.yaml对样本的 'instruction' 字段增强
- 5 使用data_juicer 根据 config_filter.yaml删除 'instruction' 字段的邮箱数据
- 6 filter_mail.py 对具有邮箱的样本过滤掉
- 7 filter_continue.py 过滤 'instruction' 包含 continue、Continue 且字符串长度小于10的example

# filter_long_token.py

描述：过滤超过1024 token 的样本

设置参数:
- model_path: 1B模型的路径
- data_path: 原始英文数据 raw_data_en.jsonl 的路径
- out_path：输出 jsonl 文件的路径

输出：
- 过滤后的 jsonl 文件

# embed.py

描述：计算每个样本的语义向量

设置参数：
- model_path：bge-base-en-v1.5 的模型路径（https://huggingface.co/ 上可下载模型参数）
- path: filter_long_token.py 的输出文件的存放路径

输出：
- embeds_with_output.npy : 计算了output的语义（我采用的是这个）
- embeds_without_output.npy : 没有包含output

# sample.py

描述：根据样本的语义相似度进行采样

设置参数：
- data_path : filter_long_token.py 的输出文件的存放路径
- embeding_path : embed.py 的输出文件的存放路径
- outfile_path : 输出 jsonl 文件路径
- filter_outfile_path ：中间 jsonl 文件的路径
- filter_embeds_outfile_path : 中间 .npy 文件的路径

输出：
- outfile_path : 输出 jsonl 文件路径
- filter_outfile_path ：中间 jsonl 文件的路径
- filter_embeds_outfile_path : 中间 .npy 文件的路径

# data_juicer config_mapper.yaml

描述：通过 fix_unicode_mapper、remove_long_words_mapper、sentence_split_mapper、whitespace_normalization_mapper 等算子对 'instruction' 字段处理

# data_juicer config_filter.yaml

描述：通过 clean_email_mapper 等算子对 'instruction' 字段处理

# filter_mail.py 

描述：由于data_juicer只能清理邮箱，自行对具有邮箱的样本过滤掉

设置参数：
- data_path_1 : config_mapper.yaml 处理后的输出路径
- data_path_2 ： config_filter.yaml 处理后的输出路径
- outfile_path ：输出 jsonl 文件路径

# filter_continue.py

描述：过滤 'instruction' 包含 continue、Continue 且字符串长度小于10的example

设置参数：
- data_path ： filter_mail.py 处理后的输出路径
- outfile_path ：输出 jsonl 文件路径