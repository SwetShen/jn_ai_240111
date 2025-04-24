# 模型下载地址
# 1. modelscope.cn 下载模型
# 2. huggingface.co 下载模型 https://huggingface.co/BAAI/bge-small-zh
# bge 词嵌入模型，用在包含中文的词嵌入
import numpy as np
# 词嵌入模型
# 作用: 将文本转换成向量
# 用在哪里?
# 1. 训练和使用语言模型的时候，但是类似 GPT 这样的模型会使用自己的词嵌入模型
# 2. 知识库中，做文本检索的时候
# 官网: https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md

# 安装: pip install -U FlagEmbedding
# pip install peft

# FlagEmbedding 依赖于 sentence_transformers，所以可以直接安装 sentence_transformers
# 示例代码:
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh')
# embeddings = model.encode(['你好', '你叫什么名字'])


from sentence_transformers import SentenceTransformer

# 构造词嵌入模型
# tokenizer: 分词器
model = SentenceTransformer(r'D:\projects\ai_models\test\bge-small-zh')

# 密集编码: 把完整的一段文本编码成一个向量，称为密集编码
encoded = model.encode('你好，吃了没？')
print(encoded)
print(type(encoded))
print(encoded.shape)

# 批量编码
encoded = model.encode([
    '你好，吃了没？',
    '你好，吃完没？'
])
print(encoded.shape)

# 相似度得分
# 点乘相似度得分
point = encoded[0].reshape(1, -1) @ encoded[1].reshape(1, -1).T
print(point)

# 距离相似度
point = np.sqrt(((encoded[0] - encoded[1]) ** 2).sum())
print(point)

# 余弦相似度得分
A = encoded[0]
B = encoded[1]
fz = np.dot(A, B)
fm = np.sqrt(np.dot(A, A)) * np.sqrt(np.dot(B, B))
point = fz / fm
print(point)

# 分词
# add_special_tokens: 是否添加特殊符号，例如: [CLS] [SEP]
# result = model.tokenizer(['你好，吃了没？'], add_special_tokens=True)
result = model.tokenizer(['你好，吃了没？'], add_special_tokens=False)
# result 是字典
# input_ids: token 的索引
print(result)

# 转换 idx list 变成 token 列表
texts = model.tokenizer.convert_ids_to_tokens(result['input_ids'][0])
print(texts)

# 可以将分词结果传入模型，进行词嵌入编码
word_vectors = model.encode(texts)
print(word_vectors.shape)

# 获取词库的大小
print(len(model.tokenizer))
print(model.tokenizer.vocab_size)

# 解码
decoded = model.tokenizer.decode(result['input_ids'][0])
print(decoded)
