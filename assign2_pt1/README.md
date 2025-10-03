# Assignment 2 Part 1: Implementing SGEMM Kernels
<!--![sgemm](https://github.com/user-attachments/assets/0239731a-c182-4e22-bb7c-cdc3f5ccadb2)-->
![naive-kernel](https://github.com/user-attachments/assets/03ab8925-0275-4b28-a3b2-a8fb10073bcf)

In this assignment you will learn how to improve a naive matmul kernel all the way to a CUBLAS-like performance kernel. For the first part of this assignment you will incrementally build up four SGEMM (FP32) kernels with different optimizations. Make use of optimizations like coalescing, register reuse and shared memory. Keep in mind, you may want to change which input matrices are transposed or not according to how you implement your access patterns in the kernels (Modify the specified lines in `test_sgemm.py` to do so.)


## To get started:
- Install the necessary packages in your environment

- Implement your kernels and your kernel wrappers in `sgemm.cu`

- Run `python3 test_sgemm.py arg` after implementing your kernel

- `usage: test_sgemm.py [-h] (-v1 | -v2 | -v3 | -v4 | -all)`



## Nsight Compute Profiling
In addition to GPU programming, one should also become familar with GPU profiling. I recommend downloading [Nsight Compute](https://developer.nvidia.com/nsight-compute) to use the UI. Answer the following short questions as you finish the implementation for each kernel. Run `ncu --set detailed python3 test_sgemm.py arg` to do the profiling in the CLI or ` ncu --set full --import-source yes --export reports/sgemm_all python3 test_sgemm.py -arg` to produce a `.ncu-rep file` to use the NCU UI on.

### Kernel 1: Naive
Look at Memory Worklord Analysis Section, how high is the Mem Bsy [%] statistic? Also take a look at Compute Worklord Analysis, how low is the SM Busy [%] statistic? What does this tell us about the naive kernel?

### Kernel 2: Coalescing
After implementing coalescing, how much did compute (SM) throughput improve by compared to naive?

### Kernel 3: Register Reuse
How many more registers per thread were used compared to the previous two kernels?

### Kernel 4: Shared memory
How much shared memory per block was allocated? Looking at the Memory Worklord Analysis Section, with all these optimizations in the end how much did memory throughput improve by compared to the naive kernel?
