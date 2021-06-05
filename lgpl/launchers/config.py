
import os
from os.path import dirname
from os.path import expanduser
REPO_DIR = dirname(dirname(dirname(os.path.realpath((__file__)))))

CODE_DIRS_TO_MOUNT = [
    REPO_DIR,
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir=expanduser('~') + '/.mujoco/',
        mount_point='/root/.mujoco',
    ),
    # dict(
    #     local_dir=REPO_DIR + '/data',
    #     mount_point='/root/language/irl/data'
    # )
]
LOCAL_LOG_DIR = REPO_DIR + 'data/local/'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    REPO_DIR + '/lgpl/launchers/run_experiment_from_doodad.py'
)
DOODAD_DOCKER_IMAGE = 'jcoreyes/irl:latest'

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
# OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/dir/from/railrl-config/'
