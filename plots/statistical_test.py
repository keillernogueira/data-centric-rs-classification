import os
import argparse
from collections import defaultdict

import numpy as np
import scipy.stats as stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_path", type=str, required=False, default='results/',
                        help="Path to the result files")
    parser.add_argument("--dataset", type=str, required=True, choices=['vaihingen', 'dfc', 'potsdam'],
                        help="Dataset to be processed")
    parser.add_argument("--alpha_level", type=float, default=0.05, help='Alpha level')
    args = parser.parse_args()

    percentages = ['1', '5', '10', '25', '50', '75', '100']

    data = defaultdict(np.array)
    baseline_data = None
    for fn in os.listdir(args.general_path):
        if args.dataset in fn:
            cur_d = np.genfromtxt(os.path.join(args.general_path, fn), delimiter='\t', dtype=float)
            if 'baseline' in fn:
                baseline_data = cur_d
            else:
                data[fn] = cur_d

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    for k, v in data.items():
        for i in range(v.shape[1]):  # columns - 1, 5, 10, 25, 50...%
            test_out = stats.ttest_rel(v[:, i], baseline_data[:, i])
            # print(k, percentages[i], test_out, v[:, i], baseline_data[:, i])
            if test_out.statistic > 0:  # and test_out.pvalue < args.alpha_level:  # better than respective baseline
                test_out_100 = stats.ttest_rel(v[:, i], baseline_data[:, -1])
                print(k, percentages[i], test_out, test_out_100)
                # if test_out_100.statistic > 0 and test_out_100.pvalue < args.alpha_level:
                #     print('-----', k, '100', test_out_100)



