import time
def timing_decorator(func):
    '''
    Decorator for timing functions
    '''
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_time//60
        print(f"Function {func.__name__} took {execution_time:.4f} minutes to execute.")
        return result
    return wrapper
