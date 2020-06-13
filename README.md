# High-Performance-Computing

Repository consists of 3 algorithms:

  1. Matrix Multiplication
  2. PI calc
  3. Bilateral
  
  
They have been processed via GPU and CPU. The results are illustrated in the tables below.
  
  ___
  
### 1 Matrix Multiplication

#### Results

*GPU time and CPU time* columns display time in seconds

*CPU/GPU* column displayes how many times GPU faster than CPU

Matrix size | GPU time | CPU time | CPU/GPU 
--- | --- | --- | --- 
256 | 0.0153 | 10.1974 | 665.4691 
512 | 0.1350 | 81.6030 | 604.0486 
1024 | 0.9650 | 687.0884 | 711.9548
2048 | 7.5562 | 5625.8987 | 744.5321

___

### 2 PI Calc 

*Task definition:* to calculate PI with Monte-Carlo algorithm via GPU and CPU

*Input:*

1. N - number of points

*Output:* 


1. Calculation time
2. PI values

#### Results

*GPU time and CPU time* columns display time in seconds

*CPU/GPU* column displayes how many times GPU faster than CPU

*Pi* column dysplays result of calculated PI

N | GPU time | CPU time | CPU/GPU | Pi  
--- | --- | --- | ---  | --- 
2^15 | 0.0006 | 0.0227 | 33.4243 | 3.1417 
2^16 | 0.0003 | 0.0425 | 115.6591 | 3.1393 
2^17 | 0.0006 | 0.0889 | 141.2439 | 3.1454 
2^18 | 0.0010 | 0.1778 | 177.6765 | 3.1390 
2^19 | 0.0017 | 0.3526 | 205.1461 | 3.1399 

___

## 3 Bilateral

*Task definition:* to make bilateral filter which is extension of Gaussian, and later process a picture with the filter both with GPU and CPU. 

#### Results

*Input:*

1. Graysclae image 
2. Sigma values - sigma r and sigma d

*Output:* 

1. Calculation time (GPU and CPU)
2. Resulting Image

sigma R | sigma D | GPU time | CPU time | CPU/GPU 
--- | --- | --- | ---  | --- 
40 | 140 |   0.3182 | 218.5789 | 686.8950 
200 | 400 |   0.3022 | 164.9899 | 545.9436

sigma R 40
sigma D 140

![img](https://github.com/thePolly/High-Performance-Computing/blob/master/r40d140.bmp)

sigma R 200
sigma D 400

![img](https://github.com/thePolly/High-Performance-Computing/blob/master/r200d400.bmp)

