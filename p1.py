inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]

# constant which is addded to the product of features and weights
bias = 2

output = 0
for x in range(len(inputs)):
    output += (inputs[x] * weights[x])
    if x == (len(inputs)-1):
        output += bias

print(output)
