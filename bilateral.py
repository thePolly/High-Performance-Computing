import numpy as np
import time
import cv2

import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule


mod = SourceModule("""
texture<unsigned int, 2, cudaReadModeElementType> tex;
__global__ void calc(unsigned int* result, const int M, const int N, const float sigma_d, const float sigma_r)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    if ((i < M) && (j < N)) {
        float h = 0;
        float kk = 0;
        for (int l = i - 1; l <= i + 1; l++){
            for (int k = j - 1; k <= j + 1; k++){
                float img1 = tex2D(tex, k, l) ;
                float img2 = tex2D(tex, i, j) ;
                float g = exp(-(pow(k - i, 2) + pow(l - j, 2)) / pow(sigma_d, 2));
                float r = exp(-pow((img1 - img2) , 2) / pow(sigma_r, 2));
                kk += g * r;
                h += g * r * tex2D(tex, k, l);
            }
        }
        result[i * N + j] = h / kk;
    }
}
""")


def cpu_calc(image, sigma_r, sigma_d):
    start_time = time.time()
    result = np.zeros(image.shape)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            kk = 0
            h = 0
            for k in range(i-1, i+2):
                for l in range(j-1, j+2):
                    g = np.exp(-((k - i) ** 2 + (l - j) ** 2) / sigma_d ** 2)
                    r = np.exp(-((image[k, l] - image[i, j])) ** 2 / sigma_r ** 2)
                    kk += g*r
                    h += g*r*image[k, l]
            result[i, j] = h / kk  
    end_time = time.time()
    cv2.imwrite('img_cpu.bmp', result)
    return end_time - start_time

def gpu_calc(image, sigma_r, sigma_d):
    N, M  = image.shape[0], image.shape[1]
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(N/block_size[0])),int(np.ceil(M/block_size[1])))
    result = np.zeros((N, M), dtype = np.uint32)
    calc = mod.get_function("calc")
    start = time.time()
    tex = mod.get_texref("tex")
    driver.matrix_to_texref(image.astype(np.uint32), tex, order="C")
    calc(driver.Out(result), np.int32(N), np.int32(M), np.float32(sigma_d), np.float32(sigma_r), block=block_size, grid=grid_size, texrefs=[tex])
    driver.Context.synchronize()
    end = time.time()
    cv2.imwrite('img_gpu.bmp', result.astype(np.uint8))
    return end - start


image = cv2.imread('img.bmp', cv2.IMREAD_GRAYSCALE)
sigma_r = 200
sigma_d = 400

cpu_time = cpu_calc(image, sigma_r, sigma_d)
gpu_time =gpu_calc(image, sigma_r, sigma_d) 

print('GPU time {}'.format(gpu_time))
print('CPU time {}'.format(cpu_time))
print('CPU/GPU {}'.format(cpu_time/gpu_time))
