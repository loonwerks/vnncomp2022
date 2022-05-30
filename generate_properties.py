# -------------------------------------------------------------------------------
# Copyright (c) Collins Aerospace Proprietary
#
# Purpose: Script to generate properties for the Remaining Useful Life (RUL)
# benchmark problem from Collins Aerospace at the 2022 VNN Competition.
#
# ITC statement: This file does not contain any export controlled technical data.
# -------------------------------------------------------------------------------

'''
The script accepts a random seed as input (for reproducibility purposes). It
also uses files from the data folder (.mat/.csv) to generate vnnlib properties.

Following types of properties are generated
  - Robustness properties (local)
  - Monotonicity properties (local)
  - If-Then properties (check that output is in a given range given input ranges)
'''

import os
import argparse
import numpy as np
import random
import scipy.io
from typing import List, Dict, Union


# --- constants
# general
WINDOW_SIZES = [20, 40]
FIRST_CI_IDX = 0
LAST_CI_IDX = 7
BOOL_FEATURES = [9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# robustness properties
DELTAS = [5, 10, 20, 40]       # percent
EPSILON = 10                   # percent
PERTURBATIONS = [2, 4, 8, 16]  # number of inputs to perturb
# monotonicity properties
ALPHA = 10                     # time units
SHIFT = [5, 10, 20]            # percent
# if-then properties
NUM_CI_RANGES = [5, 7, 9]


def generate_robustness_properties(path: str):
    '''
    Generate local robustness properties for randomly selected local points.
    Properties are generated with following parameters:

        - window size (i.e., for CNNs with different input sizes)
        - delta (input perturbation percentage)
        - epsilon (output deviation percentage)
        - number of perturbations

    For each property a vnnlib file is generated.
    '''
    for w in WINDOW_SIZES:
        mat = scipy.io.loadmat(os.path.join(path, f'test_data_w{w}.mat'))
        testdata = mat['sequences'][0]

        for d in DELTAS:
            for num_p in PERTURBATIONS:
                # take a random input sequence (window) from the test set
                idx = random.randint(0, len(testdata) - 1)
                x = testdata[idx][0]
                y = testdata[idx][1].item()

                # generate perturbation positions for the input window
                row_range = [0, x.shape[1] - 1]
                col_range = [FIRST_CI_IDX, LAST_CI_IDX]
                perturb_pos = generate_random_perturbation_pos(
                    num_p, row_range, col_range)

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

                # define variable types
                types = np.full(x.shape, 'Real', dtype='<U4')

                # 25.05.2022: Bool types are not yet supported by VNN tools
                # bool_col = np.array(['Bool' for i in range(0, x.shape[0])])
                # for t in BOOL_FEATURES:
                #     types[:, t] = bool_col

                # generate a vnnlib file
                property_name = f'robustness_{num_p}perturbations_delta{d}_w{w}.vnnlib'
                spec_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                         'specs', property_name)
                write_vnnlib_spec('robustness', spec_path, lb_in.flatten(),
                                  ub_in.flatten(), types.flatten(), lb_out, ub_out)


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


def generate_monotonicity_properties(path: str):
    '''
    Generate local monotonicity properties for randomly selected local points.
    Properties are generated with following parameters:

        - window size (i.e., for CNNs with different input sizes)
        - shift (percentage of monotonic shift for a randomly chosen feature)
        - ALPHA: admissible non-monotonicity

    For each property a vnnlib file is generated.
    '''
    for w in WINDOW_SIZES:
        mat = scipy.io.loadmat(os.path.join(path, f'test_data_w{w}.mat'))
        testdata = mat['sequences'][0]

        for s in SHIFT:
            # randomly choose a Condition Indicator (CI) to apply a monotonic shift
            ci_idx = random.randint(0, LAST_CI_IDX)

            # take a random input sequence (window) from the test set
            idx = random.randint(0, len(testdata) - 1)
            x = testdata[idx][0]
            y = testdata[idx][1].item()

            # generate monotonic shift for the randomly chosen CI
            delta_array = np.zeros(x.shape)
            delta_array[:, ci_idx] = abs(x[:, ci_idx]) * s/100

            # define input/output bounds
            lb_in = x
            ub_in = x + delta_array
            lb_out = 0
            ub_out = y + ALPHA

            # define variable types
            types = np.full(x.shape, 'Real', dtype='<U4')

            # 25.05.2022: Bool types are not yet supported by VNN tools
            # bool_col = np.array(['Bool' for i in range(0, x.shape[0])])
            # for t in BOOL_FEATURES:
            #     types[:, t] = bool_col

            # generate a vnnlib file
            property_name = f'monotonicity_CI_shift{s}_w{w}.vnnlib'
            spec_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                     'specs', property_name)
            write_vnnlib_spec('monotonicity', spec_path, lb_in.flatten(),
                              ub_in.flatten(), types.flatten(), lb_out, ub_out)


def generate_if_then_properties(path: str):
    '''
    Generate if-then properties.
    Property values have been pre-generated in Matlab. This function reads
    .mat files and picks a random property from each file. There are several
    files that correspond to different number of ranges for each feature.
    Properties with fewer ranges are harder to verify.

    NOTE: Overall, these properties seem to be the hardest.
    '''
    w = 20
    for i in NUM_CI_RANGES:
        # pick a random if-then property from the mat file
        mat = scipy.io.loadmat(
            os.path.join(path, f'if_then_properties_{i}levels.mat'))
        properties = mat['if_then'][0]
        idx = random.randint(0, len(properties) - 1)

        # define input/output bounds
        lb_in = properties[idx][1]
        ub_in = properties[idx][2]
        lb_out = properties[idx][3].item()
        ub_out = properties[idx][4].item()

        # define variable types
        types = np.full(lb_in.shape, 'Real', dtype='<U4')

        # 25.05.2022: Bool types are not yet supported by VNN tools
        # bool_col = np.array(['Bool' for i in range(0, lb_in.shape[0])])
        # for t in BOOL_FEATURES:
        #     types[:, t] = bool_col

        # generate a vnnlib file
        property_name = f'if_then_{i}levels_w{w}.vnnlib'
        spec_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 'specs', property_name)
        write_vnnlib_spec('monotonicity', spec_path, lb_in.flatten(),
                          ub_in.flatten(), types.flatten(), lb_out, ub_out)


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
        f.write("(assert (or\n")
        if lb_out is not None:
            f.write(f"\t(and (>= Y_{0} {lb_out}) ")
        if ub_out is not None:
            f.write(f"(<= Y_{0} {ub_out}))")
        # f.write(f"\t(>= Y_{1} {lb_out}) (<= Y_{1} {ub_out})\n")
        # f.write(f"(assert (>= Y_{1} {lb_out}))\n")
        # f.write(f"(assert (<= Y_{1} {ub_out}))\n")
        f.write("\n))")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUL properties generator (vnnlib)')
    parser.add_argument(
        'seed', type=int, default=0, help='random seed for property selection')
    args = parser.parse_args()

    data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    spec_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'specs')

    # set the random seed
    random.seed(args.seed)

    generate_robustness_properties(data_path)
    generate_monotonicity_properties(data_path)
    generate_if_then_properties(data_path)
