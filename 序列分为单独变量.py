# 直接对位赋值
# 任何序列（可迭代对象），都可以用赋值来分解为单独变量,但是需要元素与变量对应数量相当
p = (4, 5)
x, y = p
print(x, y)
data = ['a', 46, 'c', (1, 2, 3)]
x1, y1, z, a = data
print(x1, y1, z, a)

data = {'a', 46, 'c', (1, 2, 3)}
x1, y1, z, a = data
print(x1, y1, z, a)
# 这里[]与{}结果可能不同，因为{}是无序的，所以数据是随机赋值过去的，可以用来做随机数

s= 'hello'
a,b,c,d,e,=s
_,b1,c1,d1,_,=s
print(a,b,c,d,e)
print(b1)
# 可以用_来占位置，顶掉不想要的元素

# 任意长度的可迭代对象元素分解
# 如去除任意长度对象的第一个和最后一个元素
def drop_first_last(args):
    first,*minddle,last = args
    return
