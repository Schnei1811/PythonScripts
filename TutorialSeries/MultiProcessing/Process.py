import multiprocessing as mp
import random
import string


# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
                        string.ascii_lowercase
                        + string.ascii_uppercase
                        + string.digits)
                   for i in range(length))
    output.put(rand_str)


if __name__ == "__main__":

    random.seed(123)

    # Define an output queue
    output = mp.Queue()

    # Setup a list of processes that we want to run
    processes = [mp.Process(target=rand_string, args=(5, output)),
                 mp.Process(target=rand_string, args=(5, output)),
                 mp.Process(target=rand_string, args=(5, output)),
                 mp.Process(target=rand_string, args=(5, output))]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    results = [output.get() for p in processes]

    print(results)