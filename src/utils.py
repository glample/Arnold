import re
import os
import json
import random
import inspect
import argparse
import subprocess
import torch
from torch import optim


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(string):
    """
    Parse boolean arguments from the command line.
    """
    if string.lower() in FALSY_STRINGS:
        return False
    elif string.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. "
                                         "use 0 or 1")


def map_ids_flag(string):
    """
    Parse map IDs.
    """
    ids = string.split(',')
    assert len(ids) >= 1 and all(x.isdigit() for x in ids)
    ids = sorted([int(x) for x in ids])
    assert all(x >= 1 for x in ids) and len(ids) == len(set(ids))
    return ids


def bcast_json_list(param, length):
    """
    Broadcast an parameter into a repeated list, unless it's already a list.
    """
    obj = json.loads(param)
    if isinstance(obj, list):
        assert len(obj) == length
        return obj
    else:
        assert isinstance(obj, int)
        return [obj] * length


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", received "%s"' %
                        (str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def get_dump_path(main_dump_path, exp_name):
    """
    Create a directory to store the experiment.
    """
    assert len(exp_name) > 0
    # create the sweep path if it does not exist
    if not os.path.isdir(main_dump_path):
        subprocess.Popen("mkdir %s" % main_dump_path, shell=True).wait()
    sweep_path = os.path.join(main_dump_path, exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir %s" % sweep_path, shell=True).wait()
    # randomly generate a experiment ID
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    while True:
        folder_name = ''.join(random.choice(chars) for _ in range(10))
        dump_path = os.path.join(sweep_path, folder_name)
        if not os.path.isdir(dump_path):
            break
    # create the dump folder
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir %s" % dump_path, shell=True).wait()
    return dump_path


def set_num_threads(n):
    """
    Set the number of CPU threads.
    """
    assert n >= 1
    torch.set_num_threads(n)
    os.environ['MKL_NUM_THREADS'] = str(n)


def get_device_mapping(gpu_id):
    """
    Reload models to the associated device.
    """
    origins = ['cpu'] + ['cuda:%i' % i for i in range(8)]
    target = 'cpu' if gpu_id < 0 else 'cuda:%i' % gpu_id
    return {k: target for k in origins}
