# call 或 invoke
# python 中函数或对象可以被调用

def fn1():
    print('fn1')


class Fn2:
    def __call__(self):
        print('Fn2')


fn2 = Fn2()

fn3 = 'abc'

fn1()
fn2()

print(callable(fn1))
print(callable(fn2))
print(callable(fn3))
