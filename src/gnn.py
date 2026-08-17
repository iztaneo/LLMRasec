import torch
import torch.nn as nn

class GraphConv(nn.Module):
    """
    Capa Convolucional sobre Grafos (GCN - Kipf & Welling, 2017)
    Paridad exacta con NeuralSuite layers/graph_conv.h
    """
    def __init__(self, in_features=2, out_features=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x, adj_norm):
        # 1. Transformación de características H_linear = H_in * W
        h_linear = self.linear(x)
        # 2. Agregación de vecinos por matriz de adyacencia de grafo: H_agg = A_norm * H_linear
        h_agg = torch.matmul(adj_norm, h_linear)
        # 3. Activación
        return self.relu(h_agg)
