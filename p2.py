inputs = [1, 2, 3, 2.5]  # neurons from earlier (hidden) layer or actual input data from features 

weights_1 = [0.2, 0.8, -0.5, 1.0]
weights_2 = [0.5, -0.91, 0.26, -0.5]
weights_3 = [-0.26, -0.27, 0.17, 0.87]

# constant which is addded to the product of features and weights
bias_1 = 2
bias_2 = 3
bias_3 = 0.5

output = [inputs[0] * weights_1[0] + inputs[1] * weights_1[1] + inputs[2] * weights_1[2] + inputs[3] * weights_1[3] + bias_1,
          inputs[0] * weights_2[0] + inputs[1] * weights_2[1] + inputs[2] * weights_2[2] + inputs[3] * weights_2[3] + bias_2,
          inputs[0] * weights_3[0] + inputs[1] * weights_3[1] + inputs[2] * weights_3[2] + inputs[3] * weights_3[3] + bias_3]

# tuning weights and bias to have impact on output, the hard part of deep learning 

print(output)
