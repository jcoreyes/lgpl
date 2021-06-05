import errno
import os

import matplotlib.pyplot as plt
import numpy as np
import lgpl.utils.logger as logger
from matplotlib.pyplot import cm

_snapshot_dir = None

IMG_DIR = 'img'
VIDEO_DIR = 'video'
_snapshot_dir = None

def get_time_stamp():
    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
    return timestamp

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def set_snapshot_dir(dirname):
    global _snapshot_dir
    _snapshot_dir = dirname


def get_snapshot_dir():
    return logger.get_snapshot_dir() or _snapshot_dir or None


def logger_active():
    return get_snapshot_dir() is not None


def get_img_dir(dir_name):
    if not logger_active():
        raise NotImplementedError()
    dirname = os.path.join(get_snapshot_dir(), dir_name)
    mkdir_p(dirname)
    return dirname

def record_fig(name, dir_name, itr=None):
    if not logger_active():
        return
    if itr is not None:
        name = ('itr%d_' % itr) + name
    filename = os.path.join(get_img_dir(dir_name), name)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_traj(traj, name, marker=None, color=None, fillstyle=None):
    return plt.plot(traj[:, 0], traj[:, 1], label=name,
             marker=marker, color=color, fillstyle=fillstyle)