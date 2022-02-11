from Classes.FC import FC

inputs = [[0, 0], [1, 1], [0, 1], [1, 0]]
targets = [[1], [1], [0], [0]]
nn = FC((2, 2, 1), .5, .9)
nn.setDataSet(inputs)
nn.setTargetsSet(targets)

nn.trainFromDataInObject(1_000, 1_000, True)
# nn.trainFromDataInObjectWhileCostAboveMax(.00005, True)

print(nn.guess(inputs[0]))
print(nn.guess(inputs[1]))
print(nn.guess(inputs[2]))
print(nn.guess(inputs[3]))

# for layer in nn.getLayers():
#     print(layer.getBiases())
#     print(layer.getWeights())
