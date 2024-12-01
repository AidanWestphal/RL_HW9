import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim        

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))
            

        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))
        
    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)

class CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_dim[0], 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_small(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc_layer_neurons = 200

        self.layer1_filters = 32

        self.layer1_kernel_size = (4,4)
        self.layer1_stride = 1
        self.layer1_padding = 0

        #NB: these calculations assume:
        #1) padding is 0;
        #2) stride is picked such that the last step ends on the last pixel, i.e., padding is not used
        self.layer1_dim_h = (self.in_dim[1] - self.layer1_kernel_size[0]) / self.layer1_stride + 1
        self.layer1_dim_w = (self.in_dim[2] - self.layer1_kernel_size[1]) / self.layer1_stride + 1

        self.conv1 = nn.Conv2d(3, self.layer1_filters, self.layer1_kernel_size, stride=self.layer1_stride, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, self.layer1_kernel_size, stride=self.layer1_stride, padding=0)
        self.fc_inputs = int(1600)

        self.lin1 = nn.Linear(self.fc_inputs, self.fc_layer_neurons)

        self.lin2 = nn.Linear(self.fc_layer_neurons, self.out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)                                                                                                                     
        return x
