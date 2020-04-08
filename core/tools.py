## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import os
import socket
import re
from pytz import timezone
from datetime import datetime
import fnmatch
import itertools
import argparse
import sys
import six
import unicodedata
import json
import inspect
import tqdm
import logging
import torch
import ast
import numpy as np


def x2module(module_or_data_parallel):
    if isinstance(module_or_data_parallel, torch.nn.DataParallel):
        return module_or_data_parallel.module
    else:
        return module_or_data_parallel


# ----------------------------------------------------------------------------------------
# Comprehensively adds a new logging level to the `logging` module and the
# currently configured logging class.
# e.g. addLoggingLevel('TRACE', logging.DEBUG - 5)
# ----------------------------------------------------------------------------------------
def addLoggingLevel(level_name, level_num, method_name=None):
    if not method_name:
        method_name = level_name.lower()
    if hasattr(logging, level_name):
        raise AttributeError('{} already defined in logging module'.format(level_name))
    if hasattr(logging, method_name):
        raise AttributeError('{} already defined in logging module'.format(method_name))
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError('{} already defined in logger class'.format(method_name))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, logForLevel)
    setattr(logging, method_name, logToRoot)


# -------------------------------------------------------------------------------------------------
# Looks for sub arguments in the argument structure.
# Retrieve sub arguments for modules such as optimizer_*
# -------------------------------------------------------------------------------------------------
def kwargs_from_args(args, name, exclude=[]):
    if isinstance(exclude, str):
        exclude = [exclude]
    exclude += ["class"]
    args_dict = vars(args)
    name += "_"
    subargs_dict = {
        key[len(name):]: value for key, value in args_dict.items()
        if name in key and all([key != name + x for x in exclude])
    }
    return subargs_dict


# -------------------------------------------------------------------------------------------------
# Create class instance from kwargs dictionary.
# Filters out keys that not in the constructor
# -------------------------------------------------------------------------------------------------
def instance_from_kwargs(class_constructor, kwargs):
    argspec = inspect.getargspec(class_constructor.__init__)
    full_args = argspec.args
    filtered_args = dict([(k,v) for k,v in kwargs.items() if k in full_args])
    instance = class_constructor(**filtered_args)
    return instance


def module_classes_to_dict(module, include_classes="*", exclude_classes=()):
    # -------------------------------------------------------------------------
    # If arguments are strings, convert them to a list
    # -------------------------------------------------------------------------
    if include_classes is not None:
        if isinstance(include_classes, str):
            include_classes = [include_classes]

    if exclude_classes is not None:
        if isinstance(exclude_classes, str):
            exclude_classes = [exclude_classes]

    # -------------------------------------------------------------------------
    # Obtain dictionary from given module
    # -------------------------------------------------------------------------
    item_dict = dict([(name, getattr(module, name)) for name in dir(module)])

    # -------------------------------------------------------------------------
    # Filter classes
    # -------------------------------------------------------------------------
    item_dict = dict([
        (name,value) for name, value in item_dict.items() if inspect.isclass(getattr(module, name))
    ])

    filtered_keys = filter_list_of_strings(
        item_dict.keys(), include=include_classes, exclude=exclude_classes)

    # -------------------------------------------------------------------------
    # Construct dictionary from matched results
    # -------------------------------------------------------------------------
    result_dict = dict([(name, value) for name, value in item_dict.items() if name in filtered_keys])

    return result_dict


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def search_and_replace(string, regex, replace):
    while True:
        match = re.search(regex, string)
        if match:
            string = string.replace(match.group(0), replace)
        else:
            break
    return string


def hostname():
    name = socket.gethostname()
    n = name.find('.')
    if n > 0:
        name = name[:n]
    return name


def get_filenames(directory, match='*.*', not_match=()):
    if match is not None:
        if isinstance(match, str):
            match = [match]
    if not_match is not None:
        if isinstance(not_match, str):
            not_match = [not_match]

    result = []
    for dirpath, _, filenames in os.walk(directory):
        filtered_matches = list(itertools.chain.from_iterable(
            [fnmatch.filter(filenames, x) for x in match]))
        filtered_nomatch = list(itertools.chain.from_iterable(
            [fnmatch.filter(filenames, x) for x in not_match]))
        matched = list(set(filtered_matches) - set(filtered_nomatch))
        result += [os.path.join(dirpath, x) for x in matched]
    return result


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2str_or_none(v):
    if v.lower() == "none":
        return None
    return v


def str2dict(v):
    return ast.literal_eval(v)


def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(',')]


def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(',')]


def read_json(filename):

    def _convert_from_unicode(data):
        new_data = dict()
        for name, value in six.iteritems(data):
            if isinstance(name, six.string_types):
                name = unicodedata.normalize('NFKD', name).encode(
                    'ascii', 'ignore')
            if isinstance(value, six.string_types):
                value = unicodedata.normalize('NFKD', value).encode(
                    'ascii', 'ignore')
            if isinstance(value, dict):
                value = _convert_from_unicode(value)
            new_data[name] = value
        return new_data

    output_dict = None
    with open(filename, "r") as f:
        lines = f.readlines()
        try:
            output_dict = json.loads(''.join(lines), encoding='utf-8')
        except:
            raise ValueError('Could not read %s. %s' % (filename, sys.exc_info()[1]))
        output_dict = _convert_from_unicode(output_dict)
    return output_dict


def write_json(data_dict, filename):
    with open(filename, "w") as file:
        json.dump(data_dict, file)


def datestr():
    pacific = timezone('US/Pacific')
    now = datetime.now(pacific)
    return '{}{:02}{:02}_{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute)


def filter_list_of_strings(lst, include="*", exclude=()):
    filtered_matches = list(itertools.chain.from_iterable([fnmatch.filter(lst, x) for x in include]))
    filtered_nomatch = list(itertools.chain.from_iterable([fnmatch.filter(lst, x) for x in exclude]))
    matched = list(set(filtered_matches) - set(filtered_nomatch))
    return matched


# ----------------------------------------------------------------------------
# Writes all pairs to a filename for book keeping
# Either .txt or .json
# ----------------------------------------------------------------------------
def write_dictionary_to_file(arguments_dict, filename):
    # ensure dir
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)

    # check for json extension
    ext = os.path.splitext(filename)[1]
    if ext == ".json":

        def replace_quotes(x):
            return x.replace("\'", "\"")

        with open(filename, 'w') as file:
            file.write("{\n")
            for i, (key, value) in enumerate(arguments_dict):
                if isinstance(value, tuple):
                    value = list(value)
                if value is None:
                    file.write("  \"%s\": null" % key)
                elif isinstance(value, str):
                    value = value.replace("\'", "\"")
                    file.write("  \"%s\": \"%s\"" % (key, replace_quotes(str( value))))
                elif isinstance(value, bool):
                    file.write("  \"%s\": %s" % (key, str(value).lower()))
                else:
                    file.write("  \"%s\": %s" % (key, replace_quotes(str(value))))
                if i < len(arguments_dict) - 1:
                    file.write(',\n')
                else:
                    file.write('\n')
            file.write("}\n")
    else:
        with open(filename, 'w') as file:
            for key, value in arguments_dict:
                file.write('%s: %s\n' % (key, value))


class MovingAverage:
    postfix = "avg"

    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def add_value(self, sigma, addcount=1):
        self._sum += sigma
        self._count += addcount

    def add_average(self, avg, addcount):
        self._sum += avg*addcount
        self._count += addcount

    def mean(self):
        return self._sum / self._count


class ExponentialMovingAverage:
    postfix = "ema"

    def __init__(self, alpha=0.7):
        self._weighted_sum = 0.0
        self._weighted_count = 0
        self._alpha = alpha

    def add_value(self, sigma, addcount=1):
        self._weighted_sum = sigma + (1.0 - self._alpha)*self._weighted_sum
        self._weighted_count = 1 + (1.0 - self._alpha)*self._weighted_count

    def add_average(self, avg, addcount):
        self._weighted_sum = avg*addcount + (1.0 - self._alpha)*self._weighted_sum
        self._weighted_count = addcount + (1.0 - self._alpha)*self._weighted_count

    def mean(self):
        return self._weighted_sum / self._weighted_count


# -----------------------------------------------------------------
# Subclass tqdm to achieve two things:
#   1) Output the progress bar into the logbook.
#   2) Remove the comma before {postfix} because it's annoying.
# -----------------------------------------------------------------
class TqdmToLogger(tqdm.tqdm):
    def __init__(self, iterable=None, desc=None, total=None, leave=True,
                 file=None, ncols=None, mininterval=0.1,
                 maxinterval=10.0, miniters=None, ascii=None, disable=False,
                 unit='it', unit_scale=False, dynamic_ncols=False,
                 smoothing=0.3, bar_format=None, initial=0, position=None,
                 postfix=None,
                 logging_on_close=True,
                 logging_on_update=False):

        super(TqdmToLogger, self).__init__(
            iterable=iterable, desc=desc, total=total, leave=leave,
            file=file, ncols=ncols, mininterval=mininterval,
            maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
            unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
            smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
            postfix=postfix)

        self._logging_on_close = logging_on_close
        self._logging_on_update = logging_on_update
        self._closed = False

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='', ascii=False,
                     unit='it', unit_scale=False, rate=None, bar_format=None,
                     postfix=None, unit_divisor=1000):

        meter = tqdm.tqdm.format_meter(
            n=n, total=total, elapsed=elapsed, ncols=ncols, prefix=prefix, ascii=ascii,
            unit=unit, unit_scale=unit_scale, rate=rate, bar_format=bar_format,
            postfix=postfix, unit_divisor=unit_divisor)

        # get rid of that stupid comma before the postfix
        if postfix is not None:
            postfix_with_comma = ", %s" % postfix
            meter = meter.replace(postfix_with_comma, postfix)

        return meter

    def update(self, n=1):
        if self._logging_on_update:
            msg = self.__repr__()
            logging.logbook(msg)
        return super(TqdmToLogger, self).update(n=n)

    def close(self):
        if self._logging_on_close and not self._closed:
            msg = self.__repr__()
            logging.logbook(msg)
            self._closed = True
        return super(TqdmToLogger, self).close()


def tqdm_with_logging(iterable=None, desc=None, total=None, leave=True,
                      ncols=None, mininterval=0.1,
                      maxinterval=10.0, miniters=None, ascii=None, disable=False,
                      unit="it", unit_scale=False, dynamic_ncols=False,
                      smoothing=0.3, bar_format=None, initial=0, position=None,
                      postfix=None,
                      logging_on_close=True,
                      logging_on_update=False):

    return TqdmToLogger(
        iterable=iterable, desc=desc, total=total, leave=leave,
        ncols=ncols, mininterval=mininterval,
        maxinterval=maxinterval, miniters=miniters, ascii=ascii, disable=disable,
        unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
        smoothing=smoothing, bar_format=bar_format, initial=initial, position=position,
        postfix=postfix,
        logging_on_close=logging_on_close,
        logging_on_update=logging_on_update)


def cd_dotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), ".."))


def cd_dotdotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), "../.."))


def cd_dotdotdotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), "../../.."))


def tensor2numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    else:
        if isinstance(tensor, torch.autograd.Variable):
            tensor = tensor.data
        if tensor.dim() == 3:
            return tensor.cpu().numpy().transpose([1,2,0])
        else:
            return tensor.cpu().numpy().transpose([0,2,3,1])
