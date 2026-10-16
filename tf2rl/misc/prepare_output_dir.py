"""
MIT License

Copyright (c) 2017 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import numpy as np


def is_return_code_zero(args):
    """
    Return true if the given command's return code is zero.
    All the messages to stdout or stderr are suppressed.
    forked from https://github.com/chainer/chainerrl/blob/master/chainerrl/misc/is_return_code_zero.py
    """
    with open(os.devnull, 'wb') as FNULL:
        try:
            subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        except subprocess.CalledProcessError:
            # The given command returned an error
            return False
        except OSError:
            # The given command was not found
            return False
        return True


def is_under_git_control():
    """
    Return true iff the current directory is under git control.
    """
    return is_return_code_zero(['git', 'rev-parse'])


def prepare_output_dir(args, user_specified_dir=None, argv=None,
                       time_format='%Y%m%dT%H%M%S.%f', suffix=""):
    """
    Prepare a directory for outputting training results.
    An output directory, which ends with the current datetime string,
    is created. Then the following infomation is saved into the directory:
        args.txt: command line arguments
        command.txt: command itself
        environ.txt: environmental variables
    Additionally, if the current directory is under git control, the following
    information is saved:
        git-head.txt: result of `git rev-parse HEAD`
        git-status.txt: result of `git status`
        git-log.txt: result of `git log`
        git-diff.txt: result of `git diff`

    :param args (dict or argparse.Namespace): Arguments to save
    :param user_specified_dir (str or None): If str is specified, the output
        directory is created under that path. If not specified, it is
        created as a new temporary directory instead.
    :param argv (list or None): The list of command line arguments passed to a
        script. If not specified, sys.argv is used instead.
    :param time_format (str): Format used to represent the current datetime. The
    :param default format is the basic format of ISO 8601.
    :return: Path of the output directory created by this function (str).
    """
    if suffix != "":
        suffix = "_" + suffix
    time_str = datetime.datetime.now().strftime(time_format) + suffix
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        outdir = os.path.join(user_specified_dir, time_str)
        while os.path.exists(outdir):
            outdir += "_"
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        os.makedirs("results", exist_ok=True)
        outdir = tempfile.mkdtemp(prefix=time_str, dir="results")

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    # Save all the environment variables
    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        keys = ("CUDA_VISIBLE_DEVICES", "TF_DETERMINISTIC_OPS", "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "PYTHONHASHSEED")
        f.write(json.dumps({key: os.environ[key] for key in keys if key in os.environ}))

    # Save the command
    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    if is_under_git_control():
        # Save `git rev-parse HEAD` (SHA of the current commit)
        with open(os.path.join(outdir, 'git-head.txt'), 'wb') as f:
            f.write(subprocess.check_output('git rev-parse HEAD'.split()))

        # Save `git status`
        with open(os.path.join(outdir, 'git-status.txt'), 'wb') as f:
            f.write(subprocess.check_output('git status'.split()))

        # Save `git log`
        with open(os.path.join(outdir, 'git-log.txt'), 'wb') as f:
            f.write(subprocess.check_output('git log'.split()))

        # Save `git diff`
        with open(os.path.join(outdir, 'git-diff.txt'), 'wb') as f:
            f.write(subprocess.check_output('git diff'.split()))

    return outdir

def prep_log_tag(args, *kwargs):
    if len(kwargs) == 4 or len(kwargs) == 5:
        ret = kwargs[0]
        ret += '-v' if "eql" in kwargs[0] else '-u'
        if isinstance(kwargs[2], list):
            ret += str(kwargs[2]).replace('[','I').replace(']','I').replace(' ','').replace(',','-')
        elif isinstance(kwargs[2], tuple):
            ret += str(kwargs[2][1]).replace('[','I').replace(']','I').replace(' ', '').replace(',', '-')
            ret += '-u' + str(kwargs[2][0]).replace('[','I').replace(']','I').replace(' ', '').replace(',', '-')
        else:
            raise NotImplementedError
        if hasattr(args, "latent_units"):
            ret += '-ln' + str(args.latent_units)
            if hasattr(args, "units_decoder"):
                ret += '-dI' + args.units_decoder.replace(' ','').replace(',','-') + "I"
            elif hasattr(args, "v"):
                ret += '-dI' + str(args.v).replace(',','-') + 'I'
                ret += '-dout' + str(args.dropout).replace('.', '').replace(',', '-')
            else:
                raise NotImplementedError("dump in prep_log_tag")
            ret += '-kl' + str(args.kl_target).replace('.','')
            ret += '-nt' + str(args.n_training)
        ret += '-b' + str(args.batch_size_ail)
        ret += '-lr' + np.format_float_scientific(args.lr_ail).replace('.', '').replace('-','')
        if "eql" in kwargs[0]:
            ret += '-udp' + str(args.update_delay)
            ret += '-dout' + str(args.dropout).replace('.','').replace(',','-')
            if args.exclude:
                ret += '-eXI' + args.exclude.replace(',', '-') + 'I'
        ret += '-nois' + str(args.discriminator_noise).replace('.','')
        ret += '-epis'
        ret += 'T' if args.absorbed_state_disable else 'F'
        if hasattr(args,'state_only') and args.state_only:
            ret += '-SO'
        if args.random_epoch_n > 0:
            ret += '-rane' + str(args.random_epoch_n)
        if args.memory_capacity < 1e6:
            ret += '-mem' + np.format_float_scientific(args.memory_capacity).replace('.', '').replace('+','')
        if hasattr(args, "gp_coef") and args.gp_coef > 0:
            ret += '-gp' + str(args.gp_coef).replace('.', '')

        #Starting the policy
        ret += '_' + kwargs[1]
        ret += '-u' + str(kwargs[3]).replace('[','I').replace(']','I').replace(' ','').replace(',','-')
        ret += '-b' + str(args.batch_size)
        ret += '-lr' + np.format_float_scientific(args.lr).replace('.', '').replace('-','')
        if "sac" in kwargs[1]:
            if args.auto_alpha_disable:
                ret += '-lra' + np.format_float_scientific(args.lr_alpha).replace('.', '').replace('-','')
            else:
                ret += '-alpha' + str(args.alpha)
        ret += '_' + args.env_name.replace(':', '-')
    elif len(kwargs) == 2 or len(kwargs) == 3:
        ret = kwargs[0]
        ret += '-u' + str(kwargs[1]).replace('[', 'I').replace(']', 'I').replace(' ', '').replace(',', '-')
        ret += '-b' + str(args.batch_size)
        ret += '-lr' + np.format_float_scientific(args.lr).replace('.', '').replace('-', '')
        if "sac" in kwargs[1]:
            if args.auto_alpha_disable:
                ret += '-lra' + np.format_float_scientific(args.lr_alpha).replace('.', '').replace('-', '')
            else:
                ret += '-alpha' + str(args.alpha)
        if args.reward_dir:
            ret += '-ExtReward'
        ret += '_' + args.env_name.replace(':', '')
    else:
        raise NotImplementedError

    if len(kwargs) == 3:
        ret += '_' + str(kwargs[2])
    if len(kwargs) == 5:
        ret += '_' + str(kwargs[4])

    _ = [print(str(i) + " = < " + str(j) + " >") for i,j in args._get_kwargs()]
    return '/' + ret
