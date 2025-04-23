# 如何用RNN训练一个语言模型

RNN LSTM GRU 时序任务的预测

## 数据集的定义

一个完整的句子就是一条训练数据，例如: `how are you ? i am fine .`

数据集的输入和标签为:

- 输入: `<sos> how are you ? i am fine .`
- 标签: `how are you ? i am fine . <eos>`

# 训练流程

输入 `<sos> how are you ? i am fine .` 让模型预测 `how are you ? i am fine . <eos>`

# 测试流程

测试中 RNN 需要循环输出

第一轮: 

- 输入: `<sos> how are you ?`
- 输出: `how are you ? i`

当输出结果的最后一个字不是 `<eos>` 则继续循环

第二轮:

将上一轮输出的最后一个字拼接到输入的末尾

- 输入: `<sos> how are you ? i`
- 输出: `how are you ? i am`

当输出结果的最后一个字不是 `<eos>` 则继续循环

第三轮: ...

最后一轮:

- 输入: `<sos> how are you ? i am fine .` 
- 输出: `how are you ? i am fine . <eos>`

因为输出结果预测到了 `<eos>` 所以退出循环
