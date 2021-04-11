# GIL       Global Intepretor Lock. Memory management safeguard. Can't remove it for previous libraries

# Launches separate Python processes on alternative cores

import multiprocessing

# def spawn(num, num2):
#     print('Spawned! {} {}'.format(num, num2))
#
# if __name__ == '__main__':
#     for i in range(1xdif-100):
#         p = multiprocessing.Process(target=spawn, args=(i, i+1))            #more then one argument needs to be followed by a comma
#         p.start()
#         p.join()       #Processes waiting on each other.
#
# # In command prompt run file.
#
# from multiprocessing import Pool
#
# def job(num):
#     return num * 2
#
# if __name__ == '__main__':
#     p = Pool(processes=20)
#     data = p.map(job, range(20))
#     #Data = p.map(job, range[20])
#     #Data = p.map(job, [4])
#     data2 = p.map(job,[5,2])
#     p.close()
#     print(data)
#     print(data2)


if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()
    print(cpu_count)
    pool = multiprocessing.Pool(cpu_count)
    print(pool)
    total_tasks = 16
    tasks = range(total_tasks)
    #results = pool.map_async(work, jobs)
    #pool.close()           #Cannot submit new tasks to our pool of worker processes
    #pool.join()            #Code in __main__ must wait until all tasks are complete before continuing



from multiprocessing import Process

def func1():
  print('func1: starting')
  for i in range(10000000): pass
  print('func1: finishing')

def func2():
  print('func2: starting')
  for i in range(10000000): pass
  print('func2: finishing')

if __name__ == '__main__':
  p1 = Process(target=func1)
  p1.start()
  p2 = Process(target=func2)
  p2.start()
  p1.join()
  p2.join()