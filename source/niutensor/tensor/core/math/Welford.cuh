#ifndef __WELFORD_H__
#define __WELFORD_H__
constexpr int kWarpSize = 32;
template <typename T> __inline__ __device__ T Rsqrt(T x);

template <> __inline__ __device__ float Rsqrt<float>(float x) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
  return __frsqrt_rn(x);
#else
  return rsqrt(x);
#endif
}

template <> __inline__ __device__ double Rsqrt<double>(double x) {
  return rsqrt(x);
}
template <typename T> __inline__ __device__ T Div(T a, T b);
template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}
template <typename T>
inline __device__ void WelfordCombine(T val, T *mean, T *m2, T *count) {
  // Use Welford Online algorithem to compute mean and variance
  // For more details you can refer to:
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  *count += 1;
  T delta1 = val - *mean;
  *mean += Div(delta1, *count);
  T delta2 = val - *mean;
  *m2 += delta1 * delta2;
}
template <typename T>
inline __device__ void WelfordCombine(T b_mean, T b_m2, T b_count, T *mean,
                                      T *m2, T *count) {
  if (b_count == 0) {
    return;
  }
  T new_count = *count + b_count;
  T nb_over_n = Div(b_count, new_count);
  T delta = b_mean - *mean;
  *mean += delta * nb_over_n;
  *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
  *count = new_count;
}

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(T thread_mean, T thread_m2,
                                             T thread_count, T *mean, T *m2,
                                             T *count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpAllReduce(T thread_mean, T thread_m2,
                                                T thread_count, T *mean, T *m2,
                                                T *count) {
  WelfordWarpReduce<T, thread_group_width>(thread_mean, thread_m2, thread_count,
                                           mean, m2, count);
  *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
  *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}

// only support L2-normed
// we should consider rows_per_access and the shape of the block at the same
// time. Since a lane within a block deal with a row. Lanes belong to different
// block will become the same group.
// cols_per_thread * thread_group_width >= cols
#if 0
template <typename LOAD, typename STORE, typename ComputeType, int pack_size,
          int cols_per_thread, int thread_group_width, int rows_per_access,
          bool padding>
__global__ void LayerNormWarpImpl(DTYPE *input, DTYPE *mean, DTYPE *var,
                                  const int rows, const int cols,
                                  const DTYPE power = 1.0) {
#endif
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD input, STORE store, const int64_t rows, const int64_t cols,
                                  const double epsilon, ComputeType* mean,
                                  ComputeType* var) {
  // the id of a thread group, not the global id of a thread.
  const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  // threads that have the same blockIdx.x and the same threadIdx.y are in the
  // same group. Threads in the same group process a row(or more than one rows,
  // as indicated by rows_per_access) cooperatively.
  const int64_t num_global_thread_group = gridDim.x * blockDim.y;
  const int64_t lane_id = threadIdx.x;
  // rows_per_access means how many rows a group process
  const int64_t step = num_global_thread_group * rows_per_access;
  // num_packs means the times a thread loading data
  constexpr int num_packs = cols_per_thread / pack_size;
  ComputeType buf[rows_per_access][cols_per_thread];

  // deal all rows
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows;
       row += step) {
    ComputeType thread_mean[rows_per_access];
    ComputeType thread_m2[rows_per_access];
    ComputeType thread_count[rows_per_access];
#pragma unroll
    // in this loop, each thread finish WelfordCombine on the data they are
    // responsible for, and fill in the thread_mean, thread_m2 and thread_count
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_mean[row_id] = 0;
      thread_m2[row_id] = 0;
      thread_count[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        const int pack_offset = pack_id * pack_size;
        if (!padding || col < cols) {
          row_buf = input+(row+row_id)*cols+col;
          #if 0
          row_buf = input+(row+row_id)*cols;
          #endif
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            #if 0
            ComputeType value = row_buf[pack_offset + i];
            #endif
            ComputeType value = row_buf[i];
#if 0
            if (power != (DTYPE)1.0) {
              if (power == (DTYPE)2.0)
                value = value * value;
              else if (power == (DTYPE)0.5)
                value = sqrt(value);
              else if (power == (DTYPE)-1.0)
                value = abs(value);
              else
                value = pow(value, power);
            }
#endif
            WelfordCombine(value, thread_mean + row_id,
                           thread_m2 + row_id, thread_count + row_id);
          }
        } else {
#if 0
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            row_buf[pack_offset + i] = 0;
          }
#endif
        }
      }
    }
    // Bad name. group_xxx is better? Since a group is not always a warp.
    ComputeType warp_mean[rows_per_access];
    ComputeType warp_m2[rows_per_access];
    ComputeType warp_count[rows_per_access];
    // All threads finish all reduce cooperatively. The 0-thread withn a lane
    // will get the mean and variance of the row.
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      int global_row_id = row + row_id;
      WelfordWarpAllReduce<ComputeType, thread_group_width>(
          thread_mean[row_id], thread_m2[row_id], thread_count[row_id],
          warp_mean + row_id, warp_m2 + row_id, warp_count + row_id);
      ComputeType row_mean = warp_mean[row_id];
      ComputeType row_variance = max(Div(warp_m2[row_id], warp_count[row_id]),
                               static_cast<ComputeType>(0.0));
      if (lane_id == 0) {
        mean[global_row_id] = row_mean;
        var[global_row_id] = row_variance;
      }
    }
  }
}

template<class Func>
inline cudaError_t GetNumBlocks(Func func, int64_t block_size, size_t dynamic_smem_size,
                                int64_t max_blocks, int64_t waves, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int max_active_blocks;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, func,
                                                                    block_size, dynamic_smem_size);
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
  return cudaSuccess;
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
inline cudaError_t LaunchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols,
                                           const double epsilon, ComputeType* mean,
                                           ComputeType* inv_variance) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err =
        GetNumBlocks(LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                       thread_group_width, rows_per_access, padding>,
                     block_size, 0, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                    rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access>
inline cudaError_t DispatchLayerNormWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols,
                                                    const double epsilon, ComputeType* mean,
                                                    ComputeType* inv_variance) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                   thread_group_width, rows_per_access, false>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  } else {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                   thread_group_width, rows_per_access, true>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    if (rows % 2 == 0) {                                                                      \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    if (rows % 2 == 0) {                                                                      \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}
template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 4, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    if (rows % 2 == 0) {                                                                      \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
struct DispatchLayerNormWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance) {
    if (cols % 4 == 0) {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 4>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else if (cols % 2 == 0) {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                             const int64_t rows, const int64_t cols,
                                             const double epsilon, ComputeType* mean,
                                             ComputeType* inv_variance) {
  return DispatchLayerNormWarpImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}
#endif