

# Design a stack with a push, pop, and max function which returns the max value in the stack,
# all of which are run in 0(1) time.

#push       1   2 -> 1      3 -> 2 -> 1
#pop        3 -> 2 -> 1     2 -> 1

# store max variable. But when pop, don't know what the next highest was

# 3 -> 2 -> 1
# v    v    v
# 2    1    null

# LIFO          Stack       .insert(0, val) take from front of list .()
# FIFO          Queue       .append() take from front of list .pop()































MyStack = []
StackSize = 3
def DisplayStack():
    print("Stack currently contains:")
    for Item in MyStack:
        print(Item)
def Push(Value):
    if len(MyStack) < StackSize:
        MyStack.append(Value)
    else:
        print("Stack is full!")
def Pop():
    if len(MyStack) > 0:
        MyStack.pop()
    else:
        print("Stack is empty.")
Push(1)
Push(2)
Push(3)
DisplayStack()
input("Press any key when ready...")
Push(4)
DisplayStack()
input("Press any key when ready...")
Pop()
DisplayStack()
input("Press any key when ready...")
Pop()
Pop()
Pop()
DisplayStack()