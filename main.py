import streamlit as st
import datetime
import time

st.header('9. 파이썬 모듈')
st.subheader('9.1 모듈과 import', anchor='#6e9363cc')
st.subheader('9.2 표준 모듈의 활용',anchor='#6e9363cc')

st.header('10. 파이썬의 파일 입출력')
st.subheader('10.1 파일 입출력의 개요',anchor='#e5e0aa4a')
st.subheader('10.2 파일 출력(output)',anchor='#e5e0aa4a')
st.subheader('10.3 파일 입력(input)',anchor='#e5e0aa4a')


st.divider()

st.button("Reset", type="primary")
if st.button("Say hello"):
    st.write("Why hello there")
else:
    st.write("Goodbye")


st.download_button(
    'download',
    '1234',
    file_name='default.txt'
)

st.link_button(
    'naver',
    url = 'https://www.naver.com/'
)

st.checkbox(
    'is_decimal',
)

st.toggle(
    '넘 졸려용'
)

check = st.toggle('is_decimal')
if check:
    st.write('option is selected')


option = st.radio(
    'options',
    ['apple', 'banana', 'kiwi'],
    index = 0
)
if option == 'apple':
    st.write(1)
elif option == 'banana':
    st.write(2)
st.write(option)

option = st.selectbox(
    'options',
    ['apple', 'banana', 'kiwi'],
    index = 0
)
st.write(option)

option = st.multiselect(
    'options',
    ['apple', 'banana', 'kiwi'],
    default=['apple'],
)
st.write(option)



color = st.select_slider(
    "Select a color of the rainbow",
    options=[
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "indigo",
        "violet",
    ],
)
st.write("My favorite color is", color)

number = st.number_input("Insert a number")
st.write("The current number is ", number) # 표현범위에 오차가 있어서 사람은 무한소수로 보나 컴퓨터는 유한소수로 받아들여서 오차가 발생한다.

age = st.slider("How old are you?", 0, 130, 25)
st.write("I'm ", age, "years old")

date = st.date_input(
    "When's your birthday",
    datetime.date(1998, 9, 29) # 크롤링시 특정날짜를 긁어오는 카드 작성 가능
)
st.write("Your birthday is:", date)
st.write(type(date))

col1, col2 = st.columns([0.3, 0.7])

with col1:
    st.header('Col1')
    st.subheader('Col1-1')

    
with col2:
    st.header('Col2')
    st.subheader('Col2-2')

    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        st.header('col2_1')
        st.subheader('col2_1')

    
    with col2_2:
        st.header('col2_2')
        st.subheader('col2_2')

with st.container():
    st.write('hello world')

with st.popover('click here!'):
    st.write('hello world')
    st.write('hello world')
    st.write('hello world')
    st.write('hello world')

tab1, tab2 = st.tabs(['apple', 'banana'])

with tab1:
    st.write('tab1')
    st.write('tab1')

with tab2:
    st.write('tab2')
    st.write('tab2')


