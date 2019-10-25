import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class Conv4Net(nn.Module):
    def __init__(self):
        super(Conv4Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, nclasses)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, training=self.training, p=0.25)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, training=self.training, p=0.25)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)    
    
class MultiScaleCNN(nn.Module):
    def __init__(self):
        super(MultiScaleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=(2,2), padding_mode='zeros')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5,  padding=(2,2), padding_mode='zeros')
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=(2,2), padding_mode='zeros')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5,  padding=(2,2), padding_mode='zeros')
        
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, padding=(2,2), padding_mode='zeros')
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5,  padding=(2,2), padding_mode='zeros')
        
        self.fc1 = nn.Linear(33824, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, nclasses)

    def forward(self, x):
        # Block 1
        block1_out = F.relu(self.conv1(x))
        block1_out = F.relu(F.max_pool2d(self.conv2(block1_out),2))
        block1_out = F.dropout(block1_out)
        
        # Block 2
        block2_out = F.relu(self.conv3(block1_out))
        block2_out = F.relu(F.max_pool2d(self.conv4(block2_out),2))
        block2_out = F.dropout(block2_out)
        
        # Block 3
        block3_out = F.relu(self.conv5(block2_out))
        block3_out = F.relu(F.max_pool2d(self.conv6(block3_out),2))
        block3_out = F.dropout(block3_out)
        
        # Flatten previous blocks and concat
        block1_flat = block1_out.view(block1_out.size(0), -1)
        block2_flat = block2_out.view(block2_out.size(0), -1)
        block3_flat = block3_out.view(block3_out.size(0), -1)
        block4_out = torch.cat((block1_flat, block2_flat, block3_flat), 1)
        
        # Fully connected layers for concatenated blocks
        block4_out = F.relu(self.fc1(block4_out))
        block4_out = F.dropout(block4_out)
        
        # Another fully connected layer for good measure
        block5_out = F.relu(self.fc2(block4_out))
        block5_out = F.dropout(block5_out)
        
        # Head softmax
        final_out = self.fc3(block5_out)
        return F.log_softmax(final_out, dim=1)
    
    
class AdaptiveMultiScaleCNN(nn.Module):
    def __init__(self):
        super(AdaptiveMultiScaleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=(2,2), padding_mode='zeros')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5,  padding=(2,2), padding_mode='zeros')
        self.avgpool1 = nn.AdaptiveAvgPool2d((24, 24))
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=(2,2), padding_mode='zeros')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5,  padding=(2,2), padding_mode='zeros')
        self.avgpool2 = nn.AdaptiveAvgPool2d((12, 12))
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, padding=(2,2), padding_mode='zeros')
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5,  padding=(2,2), padding_mode='zeros')
        self.avgpool3 = nn.AdaptiveAvgPool2d((6, 6))
        
        self.fc1 = nn.Linear((32*24*24) + (64*12*12) + (128*6*6), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, nclasses)

    def forward(self, x):
        # Block 1
        block1_out = F.relu(self.conv1(x))
        block1_out = F.relu(F.max_pool2d(self.conv2(block1_out),2))
        block1_out = F.dropout(block1_out)
        block1_out = self.avgpool1(block1_out)
        
        # Block 2
        block2_out = F.relu(self.conv3(block1_out))
        block2_out = F.relu(F.max_pool2d(self.conv4(block2_out),2))
        block2_out = F.dropout(block2_out)
        block2_out = self.avgpool2(block2_out)
        
        # Block 3
        block3_out = F.relu(self.conv5(block2_out))
        block3_out = F.relu(F.max_pool2d(self.conv6(block3_out),2))
        block3_out = F.dropout(block3_out)
        block3_out = self.avgpool3(block3_out)
        
        # Flatten previous blocks and concat
        block1_flat = block1_out.view(block1_out.size(0), -1)
        block2_flat = block2_out.view(block2_out.size(0), -1)
        block3_flat = block3_out.view(block3_out.size(0), -1)
        block4_out = torch.cat((block1_flat, block2_flat, block3_flat), 1)
                
        # Fully connected layers for concatenated blocks
        block4_out = F.relu(self.fc1(block4_out))
        block4_out = F.dropout(block4_out)
        
        # Another fully connected layer for good measure
        block5_out = F.relu(self.fc2(block4_out))
        block5_out = F.dropout(block5_out)
        
        # Head softmax
        final_out = self.fc3(block5_out)
        return F.log_softmax(final_out, dim=1)    