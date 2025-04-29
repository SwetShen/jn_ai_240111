from sentence_transformers import SentenceTransformer

embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh').eval()
# 冻结参数
for p in embedding.parameters():
    p.requires_grad = False


# 分词函数
# texts: 多句话的文本列表
def tokenizer(texts, max_len=50):
    # 分词
    result = embedding.tokenizer(texts, add_special_tokens=False, max_length=max_len, padding='max_length',
                                 truncation=True, return_tensors='pt')
    ids = result['input_ids']
    padding_mask = result['attention_mask'].float()
    padding_mask[padding_mask == 0] = float('-inf')
    padding_mask[padding_mask == 1] = 0.
    return ids, padding_mask
