import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=1.0,             #takes place of x parameter
                        help='What is the first number?')
    parser.add_argument('--y', type=float, default=1.0,
                        help='What is the second number?')
    parser.add_argument('--operation', type=str, default='add',
                        help='What operation? (add, sub, mul, or div')
    args = parser.parse_args()
    sys.stdout.write(str(calc(args)))

def calc(args):
    if args.operation == 'add':
        return args.x + args.y
    elif operation == 'sub':
        return args.x - args.y
    elif operation == 'mul':
        return args.x * args.y
    elif operation == 'div':
        return args.x / args.y

if __name__ == '__main__':
    main()

# In Command Line: C:/Directory/python "name of document" --x=5 --y=2 --operation=mul

# In Command Line: C:/Directory/python "name of document" -h         displays arguments we added
