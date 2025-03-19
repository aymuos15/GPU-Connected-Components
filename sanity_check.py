import torch
import numpy as np

from skimage import measure

import cc3d
from cc3d_gpu import gpu_connected_components

from timeit import timeit

# Matrix creation code remains the same
matrix1_numpy = np.zeros((10, 35, 35))
matrix1_numpy[0, 10:20, 10:20] = 1
matrix1_numpy[0, 25:28, 10:20] = 1
matrix1_numpy[0, 10:20, 21:30] = 1
matrix1_numpy[0, 5:7, 5:7] = 1

matrix2_numpy = np.zeros((10, 35, 35))
matrix2_numpy[0, 10:20, 6:14] = 1
matrix2_numpy[0, 10:20, 15:24] = 1
matrix2_numpy[0, 25:28, 6:14] = 1
matrix2_numpy[0, 25:28, 15:24] = 1
matrix2_numpy[0, 25:27, 25:27] = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
matrix1_torch = torch.as_tensor(matrix1_numpy, dtype=torch.float32).to(device)
matrix2_torch = torch.as_tensor(matrix2_numpy, dtype=torch.float32).to(device)

# Define functions to time
def cc3d_cc():
    matrix1_cc3d = cc3d.connected_components(matrix1_numpy)
    matrix2_cc3d = cc3d.connected_components(matrix2_numpy)
    return matrix1_cc3d, matrix2_cc3d

def torch_cc():
    matrix1_cc_torch, _ = gpu_connected_components(matrix1_torch)
    matrix2_cc_torch, _ = gpu_connected_components(matrix2_torch)
    torch.cuda.synchronize()  # Ensure CUDA operations are completed
    return matrix1_cc_torch, matrix2_cc_torch

def skimage_cc():
    matrix1_skimage = measure.label(matrix1_numpy)
    matrix2_skimage = measure.label(matrix2_numpy)
    return matrix1_skimage, matrix2_skimage

# Time the functions
cc3d_time = timeit(cc3d_cc, number=100)
torch_time = timeit(torch_cc, number=100)
skimage_cc_time = timeit(skimage_cc, number=100)

# Run once to get results for comparison
matrix1_cc3d, matrix2_cc3d = cc3d_cc()
matrix1_cc_torch, matrix2_cc_torch = torch_cc()
matrix1_skimage, matrix2_skimage = skimage_cc()

print('Are the connected components equal?')
print('Matrix1 torch & cc3d:', np.allclose(matrix1_cc_torch.cpu().numpy(), matrix1_cc3d))
print('Matrix2 torch & cc3d:', np.allclose(matrix2_cc_torch.cpu().numpy(), matrix2_cc3d))
print('Matrix1 skimage & torch:', np.allclose(matrix1_skimage, matrix1_cc_torch.cpu().numpy()))
print('Matrix2 skimage & torch:', np.allclose(matrix2_skimage, matrix2_cc_torch.cpu().numpy()))