import numpy as np
import pandas as pd

from bb_util import get_test_data, visualize_2d, visualize_3d, visualize_2d_quad


if __name__ == "__main__":
	# X, y, feat_gt = get_test_data('madelon')
	# visualize_2d_quad(X, y, feat_gt, title="madelon")

	# X, y, feat_gt = get_test_data('madelon2')
	# visualize_2d_quad(X, y, feat_gt, title="madelon2")

	# X, y, feat_gt = get_test_data('madelon3')
	# visualize_2d_quad(X, y, feat_gt, title="madelon3")

	# X, y, feat_gt = get_test_data('madelon4')
	# visualize_2d_quad(X, y, feat_gt, title="madelon4")

	# X, y, feat_gt = get_test_data('hcm_1')
	# visualize_2d_quad(X, y, feat_gt, title="hcm_1")

	X, y, feat_gt = get_test_data('hcm_2')
	visualize_2d_quad(X, y, feat_gt, title="hcm_2")

	X, y, feat_gt = get_test_data('gc')
	visualize_2d_quad(X, y, feat_gt, title="gc")

	X, y, feat_gt = get_test_data('hc_bl')
	visualize_2d_quad(X, y, feat_gt, title="hc_bl", alpha=.3)

	X, y, feat_gt = get_test_data('hc2')
	visualize_2d_quad(X, y, feat_gt, title="hc2", alpha=.3)