# 숫자맞추기

n = 11
count = 0

while True:
    count += 1
    user_input = input('입력해주세요:')

    if user_input.isdecimal() != True :
        print('숫자를 입력해주세요')

    else:
        user_input = int(user_input)
        if user_input == n :
            print('정답')
            print(f'걸린횟수는 {count}번 입니다.')
            break

        elif user_input < n :
            print('up')
            
        else:
            print('down')

