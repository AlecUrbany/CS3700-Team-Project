import torch
import torch.nn as nn
import torch.nn.functional as F

# A typical training procedure for a neural network is as follows
#    - Define the neural network that has some learnable parameters or weights
#    - Iterate over a dataset of inputs
#    - Process input through the network
#    - Compute the loss (how far is the output from being correct)
#    - Propagate gradients back into the network's parameters
#    - Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

# You just have to define the forward function, 
# and the backward function (where gradients are computed) 
# is automatically defined for you using autograd

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net, "\n")

# The learnable parameters of a model are returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size(), "\n")

input = torch.randn(1, 1, 32, 32)
print(input)
output = net(input)
print(output)



