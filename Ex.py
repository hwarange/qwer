t = int(input())

for i in range(t):
    a, b, c = map(int, input().split())
    print(f'{c%a}{c//a:02}')
