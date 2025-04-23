from torch import nn


# 语言模型
class LangModel(nn.Module):
    # vocab_size: 词汇表大小
    def __init__(self, vocab_size, hidden_size=512):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    # x: one_hot 编码后的张量
    # 输出: 预测文本的概率分布
    def forward(self, x):
        y, _ = self.lstm(x)
        y = self.fc_out(y)
        return y


if __name__ == '__main__':
    from 数据集 import vocab, LangDataset

    ds = LangDataset()
    model = LangModel(len(vocab))
    inputs, labels = ds[0]
    y = model(inputs)
    print(y.shape)
    y = y.softmax(-1)
    idx = y.argmax(-1)
