#!/usr/bin/env python
"""
.. module:: phasing.py
   :platform: Unix
   :synopsis: Command line interface to CXPhasing.

.. moduleauthor:: David Vine <djvine@gmail.com>


"""

from cxphasing.CXPhasing import CXPhasing
import cxphasing.CXUtils as CXU
import argparse
import os
import errno
import pdb

def set_cxparams(new_params_file):
    """Change the cxparams file for CXPhasing.

    :param path: The path to the new CXParams file.
    :type path: str.
    :returns:  int -- the return code.
    :raises: IOError

    """
    try:
        os.symlink(new_params_file, '../cxphasing/cxparams/CXParams.py')
    except OSError, e:
        if e.errno == errno.EEXIST:
            os.remove('../cxphasing/cxparams/CXParams.py')
            os.symlink(new_params_file, '../cxphasing/cxparams/CXParams.py')
    return 0

def start_phasing():
    """ Begin the phasing process.

    :returns: int -- the return code.
    :raises: None

    """
    phasing = CXPhasing()
    phasing.setup()
    phasing.preprocessing()
    phasing.phase_retrieval()
    phasing.postprocessing()

    return 0

  

def stop_phasing():
    """ Stop the phasing process.

    :returns: int -- the return code.
    :raises: None

    """
    pass

def do_mle():
    """ Do MLE refinement only.

    :returns: int -- the return code.
    :raises: None
    """
    phasing = CXPhasing()
    phasing.maximum_likelihood_refinement()
    return 0


def main():
    """Define the command line options.

    :returns:  int -- the return code.
    :raises: AttributeError, KeyError

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--set_params', type=str, help="Set the CXParams file to use.")
    parser.add_argument('-st', '--start', help="Start phasing", dest = 'action', action = 'store_const', const=start_phasing)
    parser.add_argument('-sp', '--stop', help="Stop phasing", dest = 'action', action = 'store_const', const = stop_phasing)
    parser.add_argument('-mle', '--max_likelihood', help="Perform maximum likelihood refinement ONLY", dest = 'action', action = 'store_const', const = do_mle)

    args = parser.parse_args()

    if args.set_params:
    # check if argument is a valid file
        if os.path.exists(args.set_params):
            set_cxparams(args.set_params)
        else:
            print '{:s} is not a valid file'.format(args.set_params)
    else:
        args.action()


if __name__=='__main__': main()