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
        self.conv2 = nn.Conv2d(32, 64, self.layer1_kernel_size, stride=self.layer1_stride, padding=0)
        self.fc_inputs = int(64 * self.layer1_dim_h * self.layer1_dim_w)

        self.lin1 = nn.Linear(self.fc_inputs, self.fc_layer_neurons)

        self.lin2 = nn.Linear(self.fc_layer_neurons, self.out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.lin1(x))
        x = self.lin2(x)                                                                                                                     
        return x