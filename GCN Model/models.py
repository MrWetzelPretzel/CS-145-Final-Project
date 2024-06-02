import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        #Added these batch normalizations which were not originally implemented
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        #Added a third convolution layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)

        x = self.fc(x)

        return F.sigmoid(x)
