import matplotlib.pyplot as plt

#def function(asfs, *args=[], **kwargs={})           # args (arguments) like list, kwargs (key word arguments) like dictionary

blog_1 = 'I am so awesome.'
blog_2 = 'Cars are cool'
blog_3 = 'Aww look at my cat!!'

site_title = 'My Blog'

def argblog_posts(title, *args):
    print(title)
    for post in args: print(post)

def kwargblog_posts(title, **kwargs):
    print(title)
    for post_title, post in kwargs.items(): print(post_title, post)

def argkwargblog_posts(title, *args, **kwargs):
    print(title)
    for arg in args: print(arg)
    for post_title, post in kwargs.items(): print(post_title, post)

def graph_operation(x, y):
    print('function that graphs {} and {}'.format(str(x), str(y)))
    plt.plot(x,y)
    plt.show()

argblog_posts(site_title, blog_1)
argblog_posts(site_title, blog_1, blog_2, blog_3)          #args lets you throw in an unlimited number of arguments

kwargblog_posts(site_title, blog_1 = 'I am so awesome.', blog_2 = 'Cars are cool', blog_3 = 'Aww look at my cat!!')

argkwargblog_posts(site_title, '1', '2', '3', blog_1 = 'I am so awesome.', blog_2 = 'Cars are cool', blog_3 = 'Aww look at my cat!!' )

x1 = [1, 2, 3]
y1 = [2, 3, 1]
graph_me = [x1,y1]
graph_operation(*graph_me)