# -*- coding: utf-8 -*-

import os,sys,traceback
def import_class(import_str):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.insert(0,dirname)
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str,
                    traceback.format_exception(*sys.exc_info())))

def get_distance(import_str = "", *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)