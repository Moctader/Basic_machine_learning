import numpy as np 
import pandas as pd 
#import keras
import os
from tqdm import tqdm

import pyautogui
import time
message=10
while message>0:
    time.sleep(4)
    pyautogui.typewrite('i need you.')
    time.sleep(2)
    pyautogui.press('enter')
    message =message-1

for i in tqdm(range(int(9e6))):
    pass

# Build function
def function1 ():
    print('áhh')
    print('ahh2')

#print('this function is outside of this era')
function1()
function1()


# function declaration
def function2(x):
    return 2*x

a=function2(3)
b=function2(4)
print(a)
print(b)

def function3 (x,y):
    return x+y

j=function3(1,2)
print(j)

print('jokes')

for i in tqdm(10):
    print(i)
# Change
