import os, torch, torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from utils import visualize_graph, visualize_embedding

from torch_geometric.datasets import KarateClub

dataset = KarateClub()
# there is only one graph instance 
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')


data = dataset[0]
G = to_networkx(data, to_undirected=data.is_undirected())
visualize_graph(G, color=data.y)

class GCN(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, 
        embedding_size, out_channels
    ):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, embedding_size)
        self.classifier = Linear(embedding_size, out_channels)
        
    def forward(self, x, edge_index, training=True):
        h = self.conv1(x, edge_index).tanh()
        h = torch.nn.functional.dropout(h, p=0.2, training=training)
        h = self.conv2(h, edge_index).tanh()
        h = self.conv3(h, edge_index).tanh()
        
        output = self.classifier(h)
        return output, h
    
model = GCN(
    in_channels=dataset.num_features, hidden_channels=4, 
    embedding_size=2, out_channels=dataset.num_classes
)

_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

# see the untrained embedding
visualize_embedding(h, color=data.y)

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

def train(data):
    optimizer.zero_grad()
    output, h = model(data.x, data.edge_index)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    return loss, h

for epoch in range(1, 401):
    loss, h = train(data)
    
    if epoch == 1 or epoch % 50 == 0:
        f'Epoch: {epoch}, Loss: {loss.item():.4f}'
        # visualize_embedding(
        #     h, color=data.y, 
        #     epoch=epoch, loss=loss
        # )