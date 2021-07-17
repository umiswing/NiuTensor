/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
* All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/*
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#include "../../XDevice.h"
#include "../../XTensor.h"
#include "Normalize.h"
#include "Normalize.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
/*
normalized the data with normal distribution (kernel code). For an input x,
y = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
>> input - the input data array
>> output - the output data array
>> mean - the mean of the input
>> var - the variance of the input
>> a - the scalar
>> b - the bias
>> epsilon - a parameter
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to go over for next block
>> blockNum - how many blocks we have
*/
template<class T, TENSOR_DATA_TYPE datatype>
__global__
void KernelNormalize(T * input, T * output, T * mean, T * var,
                     T * a, T * b, T epsilon,
                     int stride, int strideNum, int blockNum)
{
    __shared__ T iMean[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ T iVar[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iBlock[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int blockSize;

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= stride * blockNum || j >= strideNum)
        return;

    if (threadIdx.y == 0) {
        iOffset[threadIdx.x] = i % stride;
        iBlock[threadIdx.x] = i / stride;
        iMean[threadIdx.x] = mean[i];
        iVar[threadIdx.x] = var[i];
        blockSize = stride * strideNum;
    }

    __syncthreads();

    int inBlockOffset = j * stride + iOffset[threadIdx.x];
    int offset = iBlock[threadIdx.x] * blockSize + inBlockOffset;

    if (datatype == DEFAULT_DTYPE) {
        output[offset] = (DTYPE)(a[inBlockOffset] * (input[offset] - iMean[threadIdx.x])) / 
                         sqrt((DTYPE)(iVar[threadIdx.x] + epsilon)) + (DTYPE)b[inBlockOffset];
    }
    else if (datatype == X_FLOAT16) {
#if __CUDA_ARCH__ >= 600
        output[offset] = __hadd(__hdiv(__hmul(a[inBlockOffset], __hsub((half)input[offset], (half)iMean[threadIdx.x])),
                         hsqrt((half)iVar[threadIdx.x] + (half)epsilon)), __float2half(b[inBlockOffset]));
#endif
    }
}

/*
normalized the data with normal distribution. For an input x,
y = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
>> input - the input tensor
>> output - the output tensor
>> dim - dimension alone which we generate the mean and variance
>> mean - the mean of the input
>> var - the variance of the input
>> a - the scalar
>> b - the bias
>> epsilon - a parameter
*/
void _CudaNormalize(const XTensor * input, XTensor * output, int dim,
                    const XTensor * mean, const XTensor * var,
                    const XTensor * a, const XTensor * b,
                    DTYPE epsilon)
{
    int stride = 1;
    int strideNum = input->dimSize[dim];
    int blockNum = 1;
    for (int i = 0; i < input->order; i++) {
        if (i > dim)
            stride *= input->dimSize[i];
        else if (i < dim)
            blockNum *= input->dimSize[i];
    }

    int cudaGridSize[3];
    int cudaBlockSize[3];

    GDevs.GetCudaThread2D(input->devID, strideNum, stride * blockNum,
                          MAX_INT, cudaGridSize, cudaBlockSize);

    dim3 blocks(cudaGridSize[1], cudaGridSize[0]);
    dim3 threads(cudaBlockSize[1], cudaBlockSize[0]);

    int devIDBackup;
    ProtectCudaDev(a->devID, devIDBackup);

    if (input->dataType == DEFAULT_DTYPE) {
        KernelNormalize <DTYPE, DEFAULT_DTYPE><< <blocks, threads >> >((DTYPE*)input->data, (DTYPE*)output->data,
                                             (DTYPE*)mean->data, (DTYPE*)var->data,
                                             (DTYPE*)a->data, (DTYPE*)b->data, epsilon,
                                             stride, strideNum, blockNum);
    }
    else if (input->dataType == X_FLOAT16) {
#ifdef HALF_PRECISION
        __half epsilon1 = __float2half(epsilon);
        KernelNormalize <__half, X_FLOAT16> <<<blocks, threads>>> ((__half*)input->data, (__half*)output->data,
                                             (__half*)mean->data, (__half*)var->data,
                                             (__half*)a->data, (__half*)b->data, epsilon1, 
                                             stride, strideNum, blockNum);
#endif
    }

    BacktoCudaDev(a->devID, devIDBackup);
}


#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)