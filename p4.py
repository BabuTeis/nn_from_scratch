import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

#  dot product computes [0]*[0] + [1]*[1] etc.
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# weights is now transposed correctly, shapes are lined up

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

'''
inputs is 3x4
weights is 3x4

3x4 with 3x4 dot product is not possible
4 != 3, thus we need to transpose
switch columns and rows, then we have
3x4 with 4x3, then 4=4 so we can perform
the dot product correctly
'''

print(layer2_outputs)
