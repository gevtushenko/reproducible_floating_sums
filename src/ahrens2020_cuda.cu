#include "reproducible_floating_accumulator.hpp"
#include "common.hpp"
#include "common_cuda.hpp"
#include "vector.hpp"
#include "nvtx3/nvtx3.hpp"

#include <iostream>
#include <random>
#include <unordered_map>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

constexpr int M = 4;            // elements per thread array

///Tests summing many numbers one at a time without a known absolute value caps
template <int block_size, class RFA_t, class T>
__global__ void kernel_1(RFA_t *result, RFA_t *partial, const T * const x, size_t N) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  RFA_t rfa;

  // first do thread private reduction
  for (auto i = tid; i < N; i+= blockDim.x * gridDim.x) rfa += x[i];

  // Compute the block-wide sum for thread 0
  reduce<block_size>(result, partial, rfa, tid);
}

template<class FloatType, class RFA_t>
void bitwise_deterministic_summation_1(RFA_t *result_d, RFA_t *partial_d, const thrust::device_vector<FloatType> &vec_d){
  nvtx3::scoped_range r{__func__};

  auto *x = reinterpret_cast<const vector_t<FloatType>*>(thrust::raw_pointer_cast(vec_d.data()));
  auto size = vec_d.size() / vector_size<FloatType>();

  kernel_1<block_size><<<grid_size, block_size>>>(result_d, partial_d, x, size);

  CHECK_CUDA(cudaDeviceSynchronize());
}

///Tests summing many numbers without a known absolute value caps
template <int block_size, int M, class T, class RFA_t>
__global__ void kernel_many(RFA_t *result, RFA_t *partial, const T * const x, size_t N) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  RFA_t rfa;

  // first do thread private reduction
  for (auto i = tid; i < N; i+= M * blockDim.x * gridDim.x) {
    T y[M] = {};
    for (auto j = 0; j < M; j++) {
      if (i + j * blockDim.x * gridDim.x < N) y[j] = x[i + j * blockDim.x * gridDim.x];
    }
    rfa.add(y, M);
  }

  // Compute the block-wide sum for thread 0
  reduce<block_size>(result, partial, rfa, tid);
}

template<class FloatType, class RFA_t>
void bitwise_deterministic_summation_many(RFA_t *result_d, RFA_t *partial_d, const thrust::device_vector<FloatType> &vec_d){
  nvtx3::scoped_range r{__func__};

  auto *x = reinterpret_cast<const vector_t<FloatType>*>(thrust::raw_pointer_cast(vec_d.data()));
  auto size = vec_d.size() / vector_size<FloatType>();

  kernel_many<block_size, M><<<grid_size, block_size>>>(result_d, partial_d, x, size);
  CHECK_CUDA(cudaDeviceSynchronize());
}

///Tests summing many numbers with a known absolute value caps
template <int block_size, int M, class T, class RFA_t, class FloatType>
__global__ void kernel_manyc(RFA_t *result, RFA_t *partial, const T * const x, size_t N, FloatType max_abs_val) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  RFA_t rfa;

  // first do thread private reduction
  for (auto i = tid; i < N; i+= M * blockDim.x * gridDim.x) {
    T y[M] = {};
    for (auto j = 0; j < M; j++) {
      if (i + j * blockDim.x * gridDim.x < N) y[j] = x[i + j * blockDim.x * gridDim.x];
    }
    rfa.add(y, M, max_abs_val);
  }

  // Compute the block-wide sum for thread 0
  reduce<block_size>(result, partial, rfa, tid);
}

template<class FloatType, class RFA_t>
void bitwise_deterministic_summation_manyc(RFA_t *result_d, RFA_t *partial_d, const thrust::device_vector<FloatType> &vec_d, const FloatType max_abs_val){
  nvtx3::scoped_range r{__func__};

  auto *x = reinterpret_cast<const vector_t<FloatType>*>(thrust::raw_pointer_cast(vec_d.data()));
  auto size = vec_d.size() / vector_size<FloatType>();

  kernel_manyc<block_size, M><<<grid_size, block_size>>>(result_d, partial_d, x, size, max_abs_val);
  CHECK_CUDA(cudaDeviceSynchronize());
}

// Kahan summation kernel
template <int block_size, class kahan_t, class FloatType>
__global__ void kernel_kahan(kahan_t *result, kahan_t *partial, const FloatType * const x, size_t N) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  kahan_t sum = {};

  // first do thread private reduction
  for (auto i = tid; i < N; i+= blockDim.x * gridDim.x) sum += x[i];

  // Compute the block-wide sum for thread 0
  reduce<block_size>(result, partial, sum, tid);
}

template<class kahan_t, class FloatType>
void kahan_summation(kahan_t *result_d, kahan_t *partial_d, const thrust::device_vector<FloatType> &vec_d){
  nvtx3::scoped_range r{__func__};

  auto *x = reinterpret_cast<const vector_t<FloatType>*>(thrust::raw_pointer_cast(vec_d.data()));
  auto size = vec_d.size() / vector_size<FloatType>();

  kernel_kahan<block_size><<<grid_size, block_size>>>(result_d, partial_d, x, size);
  CHECK_CUDA(cudaDeviceSynchronize());
}

// naive summation kernel
template <int block_size, class AccumType, class FloatType>
__global__ void kernel_simple(AccumType *result, AccumType *partial, const FloatType * const x, size_t N) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  AccumType sum = {};

  // first do thread private reduction
  for (auto i = tid; i < N; i+= blockDim.x * gridDim.x) sum += reduce(x[i]);

  // Compute the block-wide sum for thread 0
  reduce<block_size>(result, partial, sum, tid);
}

template<class AccumType, class FloatType>
void simple_summation(AccumType *result_d, AccumType *partial_d, const thrust::device_vector<FloatType> &vec_d){
  nvtx3::scoped_range r{__func__};

  auto *x = reinterpret_cast<const vector_t<FloatType>*>(thrust::raw_pointer_cast(vec_d.data()));
  auto size = vec_d.size() / vector_size<FloatType>();

  kernel_simple<block_size><<<grid_size, block_size>>>(result_d, partial_d, x, size);
  CHECK_CUDA(cudaDeviceSynchronize());
}

// Timing tests for the summation algorithms
template<class FloatType, class SimpleAccumType>
void PerformTestsOnData(thrust::host_vector<FloatType> floats) {
  // create bins on host and copy to device
  RFA_bins<FloatType> bins;
  bins.initialize_bins();
  memcpy(bin_host_buffer, &bins, sizeof(bins));
  CHECK_CUDA(cudaMemcpyToSymbol(bin_device_buffer, &bins, sizeof(bins), 0, cudaMemcpyHostToDevice));

  // rfa result buffer setup
  using RFA_t = ReproducibleFloatingAccumulator<FloatType>;
  RFA_t *rfa_result_h;
  RFA_t *rfa_result_d;
  CHECK_CUDA(cudaMallocHost(&rfa_result_h, sizeof(RFA_t)));
  CHECK_CUDA(cudaHostGetDevicePointer(&rfa_result_d, rfa_result_h, 0));

  // rfa partials
  RFA_t *rfa_partial_d;
  CHECK_CUDA(cudaMalloc(&rfa_partial_d, grid_size * sizeof(RFA_t)));

  // simple result buffer setup
  SimpleAccumType *result_h;
  SimpleAccumType *result_d;
  CHECK_CUDA(cudaMallocHost(&result_h, sizeof(SimpleAccumType)));
  CHECK_CUDA(cudaHostGetDevicePointer(&result_d, result_h, 0));

  // simple partials
  SimpleAccumType *partial_d;
  CHECK_CUDA(cudaMalloc(&partial_d, grid_size * sizeof(SimpleAccumType)));

  // kahan result buffer setup
  kahan<SimpleAccumType> *kahan_result_h;
  kahan<SimpleAccumType> *kahan_result_d;
  CHECK_CUDA(cudaMallocHost(&kahan_result_h, sizeof(kahan<SimpleAccumType>)));
  CHECK_CUDA(cudaHostGetDevicePointer(&kahan_result_d, kahan_result_h, 0));

  // simple partials
  kahan<SimpleAccumType> *kahan_partial_d;
  CHECK_CUDA(cudaMalloc(&kahan_partial_d, grid_size * sizeof(kahan<SimpleAccumType>)));

  //Very precise output

  //Get a reference value
  std::unordered_map<FloatType, uint32_t> simple_sums;
  std::unordered_map<FloatType, uint32_t> kahan_sums;

  thrust::device_vector<FloatType> floats_d = floats;
  thrust::default_random_engine g;
  bitwise_deterministic_summation_many<FloatType>(rfa_result_d, rfa_partial_d, floats_d);

  CHECK_CUDA(cudaFree(kahan_partial_d));
  CHECK_CUDA(cudaFreeHost(kahan_result_h));

  CHECK_CUDA(cudaFree(partial_d));
  CHECK_CUDA(cudaFreeHost(result_h));

  CHECK_CUDA(cudaFree(rfa_partial_d));
  CHECK_CUDA(cudaFreeHost(rfa_result_h));
}

// Use this to make sure the tests are reproducible
template<class FloatType, class SimpleAccumType>
void PerformTestsOnSineWaveData(const int N){
  thrust::host_vector<FloatType> input;
  {
    input.reserve(N);
    // Make a sine wave
    for(int i = 0; i < N; i++){
      input.push_back(std::sin(i));
    }
  }
  PerformTestsOnData<FloatType, SimpleAccumType>(input);
}

int main(){
  int N = 1 << 28;

  while (N > 1 << 20) {
    PerformTestsOnSineWaveData<float, float>(N);
    N >>= 1;
  }

  return 0;
}
