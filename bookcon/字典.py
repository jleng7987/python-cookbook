from collections import defaultdict
from collections import OrderedDict

# 创建指定内容属性的字典
d = defaultdict(list)
for key, value in pairs:
    d[key].append(value)

# 创建有序的字典,OrderedDict内部维护了一个双向链表，所以内存是普通字典的2倍
d = OrderedDict()
d['fo'] = 1
d['fo'] = 2
d['bar'] = 3


