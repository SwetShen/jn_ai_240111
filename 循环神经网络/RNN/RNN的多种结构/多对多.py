import torch
import torch.nn as nn


class RNNMany2Many(nn.Module):
    # max_iter: 输出的最大迭代次数
    # exit_token: 退出迭代的token，默认为 0，则输出 out 若全是 0，则认为模型预测到了 <EOS> 则退出循环
    def __init__(self, input_size, hidden_size, max_iter=10, eos=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_iter = max_iter
        self.eos = eos
        self.cell = nn.RNNCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    # x (N, L, input_size)
    # h (N, hidden_size)
    def forward(self, x, h=None):
        N, L, input_size = x.shape
        if h is None:
            h = torch.zeros(N, self.hidden_size)
        # 循环编码
        for j in range(L):
            h = self.cell(x[:, j], h)

        outputs = []
        # 零输入
        zero_input = torch.zeros(N, input_size)

        # 创建一个用于存储批次索引对应的结束时的长度的字典
        # key: 批次索引
        # value: 结束时的长度
        eos_map = {}

        # 循环解码
        for i in range(self.max_iter):
            # 先输出
            out = self.fc_out(h)
            # allclose 近似相等
            is_eos = torch.tensor(
                [torch.allclose(out[j], torch.tensor(self.eos, dtype=torch.float), atol=1e-6) for j in range(N)])
            # 查询非零索引
            idx = is_eos.nonzero()
            # 若存在非零索引，则代表某个批次出现了 eos 符号
            # 则将 eos 符号进行保存
            for j in idx:
                j = j.item()
                if j not in eos_map:
                    # 保存入字典中
                    eos_map[j] = i + 1

            outputs.append(out)

            # 当每个批次都出现了 eos，则跳出循环
            if len(eos_map) == N:
                break

            # 再编码
            h = self.cell(zero_input, h)

        y = torch.stack(outputs, dim=1)

        # 创建一个有效值掩码
        # mask (N, L, hidden_size)
        mask = torch.zeros_like(y)
        for k, v in eos_map.items():
            mask[k, :v] = 1

        y = y * mask

        return y, h


if __name__ == '__main__':
    model = RNNMany2Many(2, 10)
    x = torch.rand(3, 3, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
