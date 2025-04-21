def fn1(a, b):
    return a + b


# lambda 表达式是匿名的，适合简单的内容返回
# fn2 = lambda a, b: a + b
# print(fn2(1, 2))
print((lambda a, b: a + b)(1, 2))
