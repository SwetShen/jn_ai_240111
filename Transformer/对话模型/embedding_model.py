from sentence_transformers import SentenceTransformer

embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh').eval()
# 冻结参数
for p in embedding.parameters():
    p.requires_grad = False
