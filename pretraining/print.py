import numpy as np
import torch

def normalize_adj(adj):
    """Normalized an adjacent matrix by dividing each row by the sum of the row"""
    s = adj.sum(axis=-1, keepdims=True)
    s[s == 0] = 1
    result = adj / s
    return result

orig = np.load('/home/user01/aiotlab/dung_paper/GraphProject/data/mimic3/standard/code_adj.npz')
print('original adjacency matrix' )
# Example NumPy matrix
matrix = torch.load('pretraining/similarity_matrix.pt').detach().cpu().numpy()
print("Original")
print(matrix)
matrix = np.where(matrix < 0.95, 0, matrix)
print()
print("Thresholded")
print(matrix)
norm_sim = normalize_adj(matrix)
print("Normed")

print(norm_sim)

# Save the matrix to a text file
np.savetxt('pretraining/similarity_matrix.txt', matrix, fmt='%.2f', delimiter=' ')

print("Matrix saved to 'matrix.txt'")