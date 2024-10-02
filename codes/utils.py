
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G, color, figsize=(7,7)):
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(
        G, pos=nx.spring_layout(G, seed=42), 
        with_labels=False,
        node_color=color, cmap="Set2"
    )
    plt.show()
    
def visualize_embedding(
    h, color, epoch=None, 
    loss=None, figsize=(7,7)
):
    plt.figure(figsize=figsize)
    # plt.xticks([])
    # plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()