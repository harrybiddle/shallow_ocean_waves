import numpy as np
import unittest

# make it the number of simulation cells
# dictate: cell centered, staggered i/j, etc.
# concept of worldspace and interpolation
#
# This is a cell centered grid, wrapping
# around in Y, zero at X:
#      
#        --- i --->
#      .................................
#      .       .       .       .       .
#  |   .   0   .  c10  .  c11  .   0   .
#  j   .       .       .       .       .
#  |   ........-----------------........
#  V   .       |       |       |       .
#      .   0   |  c00  |  c01  |   0   .
#      .       |       |       |       .
#      ........-----------------........
#      .       |       |       |       .
#      .   0   |  c10  |  c11  |   0   .
#      .       |       |       |       .
#      ........-----------------........
#      .       .       .       .       .
#      .   0   .  c10  .  c11  .   0   .
#       .       .       .       .       .
#     .................................
#
# This is a 
#
#
#
#

def create_central_column(ni, nj):
	''' Creates an array of heights, all zero except a central column in the 
	middle of height 1 '''
	h = np.zeros((ni, nj))
	for i in range(ni // 3, 2 * ni // 3):
		for j in range(nj // 3, 2 * nj // 3):
			h[i][j] = 1.0
	return h

def create_staggered_x(ni, nj):
	return np.zeros((ni + 1, nj))

def create_staggered_y(ni, nj):
	return np.zeros((ni, nj + 1))

def ij_iterator(ni, nj):
	for i in range(0, ni):
		for j in range(0, nj):
			yield i, j

################################################################################
# Tests
################################################################################

class TestBackgroundValue(unittest.TestCase):

	def test_starting_height_is_column(self):
		h = create_central_column(9, 9)
		for i, j in ij_iterator(9, 9):
			if (3 <= i and i <= 5) and (3 <= j and j <= 5):
				self.assertEqual(1.0, h[i][j])
			else:
				self.assertEqual(0.0, h[i][j])

	def test_staggered_x_grid_size(self):
		u = create_staggered_x(10, 10)
		self.assertEqual((11, 10), u.shape)

	def test_staggered_y_grid_size(self):
		v = create_staggered_y(10, 10)
		self.assertEqual((10, 11), v.shape)		

	def test_starting_gradient(self):
		pass

################################################################################
# Main
################################################################################

if __name__ == '__main__':
    unittest.main()