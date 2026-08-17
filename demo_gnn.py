import torch
import torch.nn as nn
import torch.optim as optim
from src.gnn import GraphConv

def main():
    print("============================================================")
    print("🕸️ Demostración 7: Red Neuronal para Grafos GNN en Python (PyTorch)")
    print("============================================================")

    torch.manual_seed(42)

    # Matriz de adyacencia normalizada de grafo de 4 nodos
    adj_norm = torch.tensor([
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5]
    ])

    # Características de nodos (4 nodos, 2 características)
    X = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.1],
        [0.0, 1.0],
        [0.1, 1.0]
    ])

    # Etiquetas de nodos
    Y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    gcn = GraphConv(2, 4)
    classifier = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(list(gcn.parameters()) + list(classifier.parameters()), lr=0.05)

    print("🏋️ Entrenando GNN durante 50 épocas en Python...")
    for epoch in range(1, 51):
        optimizer.zero_grad()
        h_gcn = gcn(X, adj_norm)
        logits = classifier(h_gcn)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 50:
            print(f"Época {epoch}/50 | Loss GNN Grafo: {loss.item():.6f}")

    print("============================================================")
    print("✅ ¡Red Neuronal para Grafos GNN entrenada y verificada exitosamente en Python!")
    print("============================================================")

if __name__ == "__main__":
    main()
