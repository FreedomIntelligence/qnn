# -*- coding: utf-8 -*-

from functools import wraps
import time
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "Running the function {} takes {:.2f} seconds".format(func.__name__,delta))
        return ret
    return _deco