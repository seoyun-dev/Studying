# 내 풀이 & 교수님 풀이
from cs1robots import *

# load_world('worlds/hurdles1.wld')
# load_world('worlds/hurdles2.wld')
load_world('worlds/hurdles3.wld')

hubo = Robot()
hubo.set_trace('blue')

def turn_right():
    for _ in range(3):
        hubo.turn_left()

def jump_one_hurdle():
    hubo.turn_left()
    hubo.move()
    turn_right()
    hubo.move()
    turn_right()
    hubo.move()
    hubo.turn_left()

def move_or_jump():
    if hubo.front_is_clear():
        hubo.move()
    else:
        jump_one_hurdle()
    
while not hubo.on_beeper():
    move_or_jump()