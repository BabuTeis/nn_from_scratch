import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

#  np.dot(inputs, weights) in wrong order, but why?
output = np.dot(weights, inputs) + bias
print(output)

#  https://youtu.be/tMrbN67U9d4?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&t=895
