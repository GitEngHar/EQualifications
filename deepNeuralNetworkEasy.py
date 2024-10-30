import numpy as np

class deepNeuralNetworkEasy:
    def __init__(self):
        self.w_x = np.array([[0.46, 0.43, -0.31, 0.16],[-0.31, 0.08, 0.18, -0.52],[-0.67, 0.27, -0.19, -0.05],[0.02, 0.04, 0.10, -0.35],[-0.87, 0.65, 0.15, -0.12]])
        self.w_u1 = np.array([[-0.68, 0.01],[0.10, 0.50],[-0.10, -0.21],[-1.25, 0.50]])
        self.w_u2 = np.array([0.43, -0.19])
    def forward(self,x):
        def step(x):
            y = x > 0
            return y.astype(np.double)
        u1 = np.dot(x,self.w_x)
        z1 = step(u1)
        u2 = np.dot(z1,self.w_u1)
        z2 = step(u2)
        u3 = np.dot(z2,self.w_u2)
        y = step(u3)
        if y == 0.0:
            print("クレジット審査が通りました")
        elif y == 1.0:
            print("クレジット審査が拒否されました")

inputs_data = [[1,0,1,0,1],[0,1,0,1,0],[1,0,0,1,1]]
input_data = []
newral_network_instance = deepNeuralNetworkEasy()

for line in inputs_data:
    column = []
    for i in range(5):
        column.append(float(line[i]))
    input_data.append(column)

for x in input_data:
    newral_network_instance.forward(x)
