# -----------------------------------------------------------------------------------------
# Copyright (c) Collins Aerospace Proprietary
#
# Purpose: Script to generate properties for the Remaining Useful Life (RUL) benchmark
# problem from Collins Aerospace - to be used for the 2022 VNN competition.
#
# ITC statement: This file does not contain any export controlled technical data.
# -----------------------------------------------------------------------------------------

'''
The script accepts as inputs a test dataset
# (.mat), a table with input intervals (.csv) and a random seed (for reproducibility).
#
# Following types of properties are generated
#   - Robustness properties (local)
#   - Monotonicity properties (local)
#   - If-Then properties (check that output is in a given range given input ranges)
'''

import os
import argparse
import numpy as np
import pandas as pd
import random
import scipy.io
import shutil
from typing import List, Dict, Union


# --- constants
# general
WINDOW_SIZES = [20, 40]
FIRST_CI_IDX = 0
LAST_CI_IDX = 7
BOOL_FEATURES = [9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# robustness properties
DELTAS = [10, 40, 60, 90]      # percent
EPSILON = 10                   # percent
PERTURBATIONS = [2, 4, 8, 16]  # number of inputs to perturb
# monotonicity properties
ALPHA = 30                     # seconds
SHIFT = [5, 10]                # percent


def generate_robustness_properties(window: int, path: str):
    mat = scipy.io.loadmat(path)

    # for w in WINDOW_SIZES:
    testset = mat['sequences']

    for d in DELTAS:
        for num_p in PERTURBATIONS:
            idx = random.randint(0, len(testset[0]) - 1)
            x = testset[0][idx][0]
            y = testset[0][idx][1].item()

            # generate perturbation positions in the input window
            row_range = [0, x.shape[1] - 1]
            col_range = [FIRST_CI_IDX, LAST_CI_IDX]
            perturb_pos = generate_random_perturbation_pos(num_p, row_range, col_range)

            # generate perturbations
            delta_array = np.zeros(x.shape)
            for i in range(0, num_p):
                row = perturb_pos[i, 0]
                col = perturb_pos[i, 1]
                delta_array[row, col] = abs(x[row, col] * d/100)

            # define input/output bounds
            lb_in = x - delta_array
            ub_in = x + delta_array
            lb_out = y * (1 - EPSILON/100)
            ub_out = y * (1 + EPSILON/100)

            # variable types
            types = np.full(x.shape, 'Real', dtype='<U4')
            bool_col = np.array(['Bool' for i in range(0, x.shape[1])])
            for t in BOOL_FEATURES:
                types[:, t] = bool_col

            # generate a vnnlib file
            property_name = f'robustness_{num_p}_perturbations_delta_{d}.vnnlib'
            spec_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                     'specs', property_name)
            write_vnnlib_spec('robustness', spec_path, lb_in.flatten(), ub_in.flatten(),
                              types.flatten(), lb_out, ub_out)


def generate_random_perturbation_pos(n, range_x, range_y):

    def generate_random_pair(x, y):
        return np.array([[random.randint(x[0], x[1]),
                         random.randint(y[0], y[1])]])

    pairs = generate_random_pair(range_x, range_y)
    count = 1
    while True:
        if count == n:
            return pairs
        else:
            new_pair = generate_random_pair(range_x, range_y)
            exists = (pairs == new_pair).all(axis=1).any()
            if ~exists:
                pairs = np.vstack([pairs, new_pair])
                count += 1


def generate_monotonicity_properties(window: int, path: str):
    mat = scipy.io.loadmat(path)

    # for w in WINDOW_SIZES:
    testset = mat['sequences']

    for s in SHIFT:
        ci_idx = random.sample(range(LAST_CI_IDX+1), 4)
        for i in ci_idx:
            idx = random.randint(0, len(testset[0]) - 1)
            x = testset[0][idx][0]
            y = testset[0][idx][1].item()

            # generate monotinic shift for a random CI
            delta_array = np.zeros(x.shape)
            delta_array[:, i] = abs(x[:, i]) * s/100

            # define input/output bounds
            lb_in = x
            ub_in = x + delta_array
            lb_out = y + ALPHA

            # variable types
            types = np.full(x.shape, 'Real', dtype='<U4')
            bool_col = np.array(['Bool' for i in range(0, x.shape[1])])
            for t in BOOL_FEATURES:
                types[:, t] = bool_col

            # generate a vnnlib file
            property_name = f'monotonicity_CI_{i}_shift_{s}.vnnlib'
            spec_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                     'specs', property_name)
            write_vnnlib_spec('monotonicity', spec_path, lb_in.flatten(), ub_in.flatten(),
                              types.flatten(), lb_out)


def generate_if_then_properties(window: int, path: str):
    pass


def write_vnnlib_spec(
        prop_type: str,
        path: str,
        lb_in: List[float],
        ub_in: List[float],
        input_types: Union[List[str], np.array],
        lb_out: int = None,
        ub_out: int = None,
        ):
    '''
    Generate a vnnlib specification for the property using lb/ub
    :arg prop_type
    :arg path:
    :arg lb_in:
    :arg ub_in:
    :arg input_types
    :arg lb_out:
    :arg ub_out:
    :return:
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:

        f.write(f"; RUL {prop_type} property.\n")

        # input variables.
        f.write(f"; Input variables:\n")
        f.write("\n")
        for i in range(len(lb_in)):
            f.write(f"(declare-const X_{i} {input_types[i]})\n")
        f.write("\n")

        # output variables.
        f.write(f"; Output variables:\n")
        f.write("\n")
        f.write(f"(declare-const Y_{1} Real)\n")
        f.write("\n")

        # input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(len(lb_in)):
            f.write(f"(assert (>= X_{i} {lb_in[i]}))\n")
            f.write(f"(assert (<= X_{i} {ub_in[i]}))\n")
            f.write("\n")
        f.write("\n")

        # output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (and\n")
        if lb_out is not None:
            f.write(f"\t(>= Y_{1} {lb_out})")
        if ub_out is not None:
            f.write(f"(<= Y_{1} {ub_out})")
        # f.write(f"\t(>= Y_{1} {lb_out}) (<= Y_{1} {ub_out})\n")
        # f.write(f"(assert (>= Y_{1} {lb_out}))\n")
        # f.write(f"(assert (<= Y_{1} {ub_out}))\n")
        f.write("\n))")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUL properties generator (vnnlib)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--testset', type=str, help='test dataset (.mat)')
    parser.add_argument('--ranges', type=str, help='table with input ranges (.csv)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for property selection')
    # parser.add_argument('--window', type=int, default=20, help='size of the time window for CNN')
    args = parser.parse_args()

    # set the random seed
    random.seed(56756747)

    path = 'D:/003_ML_Assurance/500_FM/vnn2022/testset.mat'
    spec_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'specs')
    try:
        shutil.rmtree(spec_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    generate_robustness_properties(20, path)
    generate_monotonicity_properties(20, path)
    # generate_if_then_properties(args.window, args.ranges)
