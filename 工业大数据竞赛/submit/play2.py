from random import randint


def rand_list(**key):
    num_list = []
    while True:
        res = randint(key['start'], key['end'])
        if res not in num_list:
            num_list.append(res)
        if len(num_list) == key['size']:
            break
    num_list.sort()
    return num_list


def splice(num_list):
    string = ''
    for n in num_list:
        n = str(n)
        if len(n) == 1:
            n = '0' + n
        string += n + ','
    return string[:-1]


def main():
    while True:
        try:
            num = int(input('机选几注：'))
        except Exception:
            print('输入有误,重新输入!')
            continue
        if num == 0:
            break
        for i in range(0, num):
            num1_list = rand_list(start=1, end=35, size=5)
            num2_list = rand_list(start=1, end=12, size=2)
            str1 = splice(num1_list)
            str2 = splice(num2_list)
            print(str1 + '|' + str2)


if __name__ == '__main__':
    main()