import numpy as np
import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule
import sys
import time


mod = SourceModule("""
		__global__ void calc(double *x, double *y,  double *res, int* N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread id
	int threadCount = gridDim.x * blockDim.x;
	int tmp = 0;
	for (int i = idx; i < N[0]; i += threadCount) {
		if (x[i]*x[i] + y[i]*y[i] < 1) {
			tmp++;
		}
	}
  	tmp = tmp * 4/N[0];
 	atomicAdd(res, tmp);
}
""")

def cpu_calc(n, x, y):
  start_time = time.time()
  tmp = 0
  for i in range(n):
	  if x[i]*x[i] + y[i]*y[i] < 1:
		  tmp += 1
  res = tmp * 4/n
  end_time = time.time()
  return_array = np.array([res, end_time - start_time])
  return return_array


def gpu_calc(n, x, y):
  block_size = (256, 1, 1)
  grid_size = (int(n / (128 * block_size[0])), 1)
  calc = mod.get_function("calc")
  res = np.zeros(1)
  M = np.array([n])
  start_time = time.time()
  calc(driver.In(x), driver.In(y), driver.Out(res), driver.In(M), block = block_size, grid = grid_size)
  driver.Context.synchronize()
  end_time = time.time()
  return_array = np.array([res, end_time - start_time])
  return return_array


N = np.array([2**15, 2**16, 2**17, 2**18, 2**19])

for i in range(len(N)):
  n = N[i]
  x = np.random.random(n)
  y = np.random.random(n)

  cpu_time = cpu_calc(n, x, y)
  gpu_time = gpu_calc(n, x, y)

  time_g = gpu_time[1]
  time_c = cpu_time[1]
  res_g = gpu_time[0]
  res_c = cpu_time[0]

  print('Time of GPU {}'.format(time_g))
  print('Time of CPU {}'.format(time_c))
  print('Time of CPU/GPU {}'.format(time_c/time_g))
  print('Res of CPU {}'.format(res_c))
