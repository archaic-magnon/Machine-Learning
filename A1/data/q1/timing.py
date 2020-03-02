import time


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__ + " took " + str(end - start) + " s")
        return result
    return wrapper