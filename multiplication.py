import numpy as np
import time
import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule

#algorithm of multiplication in cuda
mod = SourceModule("""
	__global__ void mult(double* a, double* b, double* c, int* N){
		const int row = blockIdx.y * blockDim.y + threadIdx.y;
		const int column = blockIdx.x * blockDim.x + threadIdx.x;
		for(int i = 0; i < N[0]; i++){
			c[row * N[0] + column] += a[row * N[0] + i] * b[i * N[0] + column];
		}
	}
""")

#gpu matrix calculation
def cuda_calc(n,a,b,c):
  block_size = (2, 2, 1)
  grid_size = (int((n + block_size[0] - 1) / 2), int((n + block_size[1] - 1) / 2))
  mult = mod.get_function("mult")
  M = np.array([n])

  start_time = time.time()
  mult(driver.In(a), driver.In(b), driver.Out(c), driver.In(M), block = block_size, grid = grid_size)
  driver.Context.synchronize()
  end_time = time.time()

  return end_time - start_time

#cpu matrix calculation
def cpu_calc(n, a, b):
  start_time = time.time()
  matrix = np.zeros((n, n))
  for i in range(n):
	  for j in range(n):
		  for k in range(n):
			  matrix[i, j] += a[i, k] * b[k, j]
  end_time = time.time()

  return end_time - start_time

#matrix sizes
N = np.array([256, 512, 1024, 2048])


for i in range(len(N)):
  n = N[i]
  a = b = np.random.randn(n, n)
  c = np.zeros((n, n))

  gpu_time = cuda_calc(n,a,b,c)
  cpu_time = cpu_calc(n,a,b)

  print(' Matrix size {} \n GPU multiplication time {}'.format(n, gpu_time))
  print(' CPU multiplication time {}'.format(cpu_time))
  print(' CPU / GPU time {}'.format(cpu_time/gpu_time))
