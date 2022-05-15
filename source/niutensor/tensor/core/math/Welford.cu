/*
Use the Welford algorithm to calculate the mean
and variance along a dimension of the tensor
>> input - the input tensor
>> mean - the tensor store the mean
>> var - the tensor store the variance
>> dim - the dimension where the Welford is performed on
*/
#include "Welford.h"
#include "../../XDevice.h"
#include "../../XTensor.h"
#include "Welford.cuh"
#include "../../XName.h"
namespace nts {

void Welford(const XTensor &input, XTensor &mean, XTensor &var, int dim) {

  CheckNTErrors(dim >= 0 && dim < input.order, "Illegal dimension to reduce!");

  if (!mean.isInit || !XTensor::IsReduceShaped(&input, &mean, dim)) {
    int order = input.order - 1;
    int *dimSize = new int[order];
    for (int i = 0; i < order; i++) {
      if (i < dim)
        dimSize[i] = input.dimSize[i];
      else if (i >= dim)
        dimSize[i] = input.dimSize[i + 1];
    }

    float dr = (!input.isSparse) ? 1.0F : input.denseRatio;
    InitTensorV2(&mean, order, dimSize, input.dataType, dr, input.devID,
                 input.mem);

    /* destroy variables */
    delete[] dimSize;
  }
  if (!var.isInit || !XTensor::IsReduceShaped(&input, &var, dim)) {
    int order = input.order - 1;
    int *dimSize = new int[order];
    for (int i = 0; i < order; i++) {
      if (i < dim)
        dimSize[i] = input.dimSize[i];
      else if (i >= dim)
        dimSize[i] = input.dimSize[i + 1];
    }

    float dr = (!input.isSparse) ? 1.0F : input.denseRatio;
    InitTensorV2(&var, order, dimSize, input.dataType, dr, input.devID,
                 input.mem);

    /* destroy variables */
    delete[] dimSize;
  }
  mean.SetTMPFlag();
  var.SetTMPFlag();
  int stride = 1;
  int strideNum = input.dimSize[dim];
  int blockSize = 1;
  int blockNum = 1;
  for (int i = 0; i < input.order; i++) {
    if (i < dim)
      blockNum *= input.dimSize[i];
    else if (i > dim)
      stride *= input.dimSize[i];
  }
  blockSize = stride * strideNum;
  // view the input tensor as a 2-order tensor with shape (blockNum,
  // strideNum*stride)
  constexpr int thread_group_width = kWarpSize;
  const int cols_per_thread =
      (strideNum * stride + thread_group_width - 1) / thread_group_width;
  constexpr int rows_per_access = 1;
  constexpr int pack_size = 1;

  int devID = input.devID;
  int devIDBackup;
  ProtectCudaDev(devID, devIDBackup);
  #if 0
  cudaStream_t Stream;
  cudaStreamCreate(&Stream);
  #endif
  DispatchLayerNormWarpImpl<float *, int, float>(
      0, (float *)input.data, 1, blockNum, stride * strideNum, 1,
      (float *)mean.data, (float *)var.data);
  #if 0
  cudaStreamDestroy(Stream);
  #endif
#if 0
  _Welford<pack_size, 1, thread_group_width, rows_per_access, true>
      <<<1, 1>>>((float *)input.data, (float *)mean.data, (float *)var.data,
                 blockNum, stride * strideNum);
#endif
  BacktoCudaDev(devID, devIDBackup);

  if (input.enableGrad) {
    /* tensor connections */
    XLink::MakeLink(&input, NULL, &mean, REDUCE_REDUCEMEAN);
    XLink::AddParamToHeadInt(&mean, dim);

    XLink::MakeLink(&input, &mean, &var, REDUCE_REDUCEVARIANCE);
    XLink::AddParamToHeadInt(&var, dim);
  }
}
} // namespace nts
