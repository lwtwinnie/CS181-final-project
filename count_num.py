# 读取文件内容
file_path = 'wins_log.txt'  # 替换为你的文件路径
with open(file_path, 'r') as file:
    content = file.read()

# 统计数字出现的次数
digit_count = {}
for digit in content:
    if digit.isdigit():
        digit = int(digit)
        digit_count[digit] = digit_count.get(digit, 0) + 1

# 输出结果
for digit, count in digit_count.items():
    print(f'Digit {digit}: {count} times')
