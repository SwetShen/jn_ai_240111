"""
        输入特征  输出标签
模型1    0.6       1
模型2    0.3       0

上述的输入特征0.6、0.3都是由w,b（神经网络）得到的
"""
import math

P = 0.3
x = 0
output = x * math.log(P) + (1 - x) * math.log(1 - P)
print(output)  # 通过极大似然值公式计算出来的值大多都是负数，因此需要加符号
