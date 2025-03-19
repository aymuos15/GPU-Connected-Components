import torch
import numpy as np

from skimage import measure

import cc3d
from cc3d_gpu import gpu_connected_components

from timeit import timeit

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_matrix(size):
    matrix = np.zeros((50, size, size))
    matrix[0, size//4:size//2, size//4:size//2] = 1
    matrix[0, 3*size//4:7*size//8, size//4:size//2] = 1
    matrix[0, size//4:size//2, 5*size//8:3*size//4] = 1
    return matrix

def torch_cc(matrix):
    result = gpu_connected_components(matrix)
    torch.cuda.synchronize()
    return result

def torch_cc_nosync(matrix):
    return gpu_connected_components(matrix)

def cc3d_cc(matrix):
    return cc3d.connected_components(matrix)

def skimage_cc(matrix):
    return measure.label(matrix)

def numpy_cc(matrix):
    # Convert numpy array to torch tensor and move to device
    matrix_torch = torch.as_tensor(matrix, dtype=torch.float32).to(device)
    return gpu_connected_components(matrix_torch)

sizes = [32, 64, 128, 256, 512, 1024, 1200]
numpy_times = []
torch_times = []
torch_nosync_times = []
cc3d_times = []
skimage_cc_times = []

runs = 10

for size in sizes:
    matrix = create_matrix(size)
    matrix_torch = torch.as_tensor(matrix, dtype=torch.float32).to(device)
    
    torch_time = timeit(lambda: torch_cc(matrix_torch), number=runs) / runs #GPU | Torch
    torch_nosync_time = timeit(lambda: torch_cc_nosync(matrix_torch), number=runs) / runs #GPU | Torch
    numpy_time = timeit(lambda: numpy_cc(matrix), number=runs) / runs #GPU | Numpy
    cc3d_time = timeit(lambda: cc3d_cc(matrix), number=runs) / runs #CPU | Numpy
    skimage_cc_time = timeit(lambda: skimage_cc(matrix), number=runs) / runs #CPU | Numpy
    
    torch_times.append(torch_time)
    torch_nosync_times.append(torch_nosync_time)
    cc3d_times.append(cc3d_time)
    skimage_cc_times.append(skimage_cc_time)
    numpy_times.append(numpy_time)

plt.figure(figsize=(12, 6))
plt.plot(sizes, torch_times, marker='s', label='PyTorch')
plt.plot(sizes, torch_nosync_times, marker='s', label='PyTorch (no sync)')
plt.plot(sizes, cc3d_times, marker='^', label='cc3d')
plt.plot(sizes, skimage_cc_times, marker='o', label='skimage')
plt.plot(sizes, numpy_times, marker='x', label='Numpy')
plt.title('Execution Time vs Matrix Size')
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds)')
plt.legend()
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True)
plt.savefig('connected_components_comparison_line_graph.png')
# plt.show()  # Removed to avoid displaying

# Create separate plots for each method
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 6))

ax1.plot(sizes, skimage_cc_times, marker='o', color='blue')
ax1.set_title('Ski-image')
ax1.set_xlabel('Matrix Size')
ax1.set_ylabel('Time (seconds)')
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.grid(True)

ax2.plot(sizes, torch_times, marker='s', color='orange')
ax2.set_title('PyTorch')
ax2.set_xlabel('Matrix Size')
ax2.set_ylabel('Time (seconds)')
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.grid(True)

ax3.plot(sizes, cc3d_times, marker='^', color='green')
ax3.set_title('cc3d')
ax3.set_xlabel('Matrix Size')
ax3.set_ylabel('Time (seconds)')
ax3.set_xscale('log', base=2)
ax3.set_yscale('log')
ax3.grid(True)

ax4.plot(sizes, numpy_times, marker='x', color='red')
ax4.set_title('Numpy')
ax4.set_xlabel('Matrix Size')
ax4.set_ylabel('Time (seconds)')
ax4.set_xscale('log', base=2)
ax4.set_yscale('log')
ax4.grid(True)

ax5.plot(sizes, torch_nosync_times, marker='o', color='purple')
ax5.set_title('PyTorch (no sync)')
ax5.set_xlabel('Matrix Size')
ax5.set_ylabel('Time (seconds)')
ax5.set_xscale('log', base=2)
ax5.set_yscale('log')

plt.tight_layout()
plt.savefig('connected_components_comparison_separate_plots.png')
# plt.show()  # Removed to avoid displaying