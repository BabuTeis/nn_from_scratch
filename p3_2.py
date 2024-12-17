import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

#  dot product computes [0]*[0] + [1]*[1] etc.
output = np.dot(weights, inputs) + biases
print(output)

#  https://youtu.be/tMrbN67U9d4?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&t=895
