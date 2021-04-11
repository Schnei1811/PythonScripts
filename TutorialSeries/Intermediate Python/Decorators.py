from functools import wraps

def add_wrapping_with_style(style):
    def add_wrapping(item):
        @wraps(item)                    #returns original new_gpu function instead
        def wrapped_item():
            return 'a {} wrapped up box of {}'.format(style,str(item()))
        return wrapped_item
    return add_wrapping


def add_wrapping(item):
    @wraps(item)  # returns original new_gpu function instead
    def wrapped_item():
        return 'a wrapped up box of {}'.format(str(item()))
    return wrapped_item

@add_wrapping
def new_gpu():
    return 'a new Tesla P100 GPU'

@add_wrapping_with_style('horribly')
@add_wrapping_with_style('beautifully')
def new_bicycle():
    return 'a new bicycle'

print(new_gpu())
print(new_bicycle())

print(new_gpu.__name__)