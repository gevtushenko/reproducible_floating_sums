#include "reproducible_floating_accumulator.hpp"
#include "common.hpp"
#include "common_cuda.hpp"
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
template <class FloatType, int block_size, class RFA_t>
__global__ void kernel_1(RFA_t *result, RFA_t *partial, const FloatType * const x, size_t N) {
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
  kernel_1<FloatType, block_size><<<grid_size, block_size>>>(result_d, partial_d, thrust::raw_pointer_cast(vec_d.data()), vec_d.size());
  CHECK_CUDA(cudaDeviceSynchronize());
}

///Tests summing many numbers without a known absolute value caps
template <int block_size, int M, class FloatType, class RFA_t>
__global__ void kernel_many(RFA_t *result, RFA_t *partial, const FloatType * const x, size_t N) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  RFA_t rfa;

  // first do thread private reduction
  for (auto i = tid; i < N; i+= M * blockDim.x * gridDim.x) {
    FloatType y[M] = {};
    for (auto j = 0; j < M; j++) {
      y[j] = (i + j * blockDim.x * gridDim.x) < N ? x[i + j * blockDim.x * gridDim.x] : 0.0;
    }
    rfa.add(y, M);
  }

  // Compute the block-wide sum for thread 0
  reduce<block_size>(result, partial, rfa, tid);
}

template<class FloatType, class RFA_t>
void bitwise_deterministic_summation_many(RFA_t *result_d, RFA_t *partial_d, const thrust::device_vector<FloatType> &vec_d){
  nvtx3::scoped_range r{__func__};
  kernel_many<block_size, M, FloatType><<<grid_size, block_size>>>(result_d, partial_d, thrust::raw_pointer_cast(vec_d.data()), vec_d.size());
  CHECK_CUDA(cudaDeviceSynchronize());
}

///Tests summing many numbers with a known absolute value caps
template <int block_size, int M, class FloatType, class RFA_t>
__global__ void kernel_manyc(RFA_t *result, RFA_t *partial, const FloatType * const x, size_t N, FloatType max_abs_val) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  RFA_t rfa;

  // first do thread private reduction
  for (auto i = tid; i < N; i+= M * blockDim.x * gridDim.x) {
    FloatType y[M] = {};
    for (auto j = 0; j < M; j++) {
      y[j] = (i + j * blockDim.x * gridDim.x) < N ? x[i + j * blockDim.x * gridDim.x] : 0.0;
    }
    rfa.add(y, M, max_abs_val);
  }

  // Compute the block-wide sum for thread 0
  reduce<block_size>(result, partial, rfa, tid);
}

template<class FloatType, class RFA_t>
void bitwise_deterministic_summation_manyc(RFA_t *result_d, RFA_t *partial_d, const thrust::device_vector<FloatType> &vec_d, const FloatType max_abs_val){
  nvtx3::scoped_range r{__func__};
  kernel_manyc<block_size, M, FloatType><<<grid_size, block_size>>>(result_d, partial_d, thrust::raw_pointer_cast(vec_d.data()), vec_d.size(), max_abs_val);
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
  kernel_kahan<block_size, kahan_t, FloatType><<<grid_size, block_size>>>(result_d, partial_d, thrust::raw_pointer_cast(vec_d.data()), vec_d.size());
  CHECK_CUDA(cudaDeviceSynchronize());
}

// naive summation kernel
template <int block_size, class AccumType, class FloatType>
__global__ void kernel_simple(AccumType *result, AccumType *partial, const FloatType * const x, size_t N) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  AccumType sum = {};

  // first do thread private reduction
  for (auto i = tid; i < N; i+= blockDim.x * gridDim.x) sum += x[i];

  // Compute the block-wide sum for thread 0
  reduce<block_size>(result, partial, sum, tid);
}

template<class AccumType, class FloatType>
void simple_summation(AccumType *result_d, AccumType *partial_d, const thrust::device_vector<FloatType> &vec_d){
  nvtx3::scoped_range r{__func__};
  kernel_simple<block_size, AccumType, FloatType><<<grid_size, block_size>>>(result_d, partial_d, thrust::raw_pointer_cast(vec_d.data()), vec_d.size());
  CHECK_CUDA(cudaDeviceSynchronize());
}

// Timing tests for the summation algorithms
template<class FloatType, class SimpleAccumType>
FloatType PerformTestsOnData(
  const int TESTS,
  thrust::host_vector<FloatType> floats //Make a copy so we use the same data for each test
){
  nvtx3::scoped_range r{__func__};

  Timer time_deterministic_1;
  Timer time_deterministic_many;
  Timer time_deterministic_manyc;
  Timer time_kahan;
  Timer time_simple;

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
  std::cout.precision(std::numeric_limits<FloatType>::max_digits10);
  std::cout<<std::fixed;

  std::cout<<"'1ata' tests summing many numbers one at a time without a known absolute value caps"<<std::endl;
  std::cout<<"'many' tests summing many numbers without a known absolute value caps"<<std::endl;
  std::cout<<"'manyc' tests summing many numbers with a known absolute value caps\n"<<std::endl;

  std::cout<<"Floating type                        = "<<typeid(FloatType).name()<<std::endl;
  std::cout<<"Simple summation accumulation type   = "<<typeid(SimpleAccumType).name()<<std::endl;

  //Get a reference value
  std::unordered_map<FloatType, uint32_t> simple_sums;
  std::unordered_map<FloatType, uint32_t> kahan_sums;

  bitwise_deterministic_summation_1<FloatType>(rfa_result_d, rfa_partial_d, floats);
  const auto ref_val = rfa_result_h->conv();
  const auto kahan_ldsum = serial_kahan_summation<long double>(floats);
  long double ref_diff = std::abs(ref_val - kahan_ldsum);
  long double simple_max_diff = 0.0;
  long double kahan_max_diff = 0.0;

  thrust::device_vector<FloatType> floats_d = floats;
  thrust::default_random_engine g;
  for(int test=0;test<TESTS;test++){
    thrust::shuffle(floats_d.begin(), floats_d.end(), g);

    time_deterministic_1.start();
    bitwise_deterministic_summation_1<FloatType>(rfa_result_d, rfa_partial_d, floats_d);
    const auto my_val_1 = rfa_result_h->conv();
    time_deterministic_1.stop();
    if(ref_val!=my_val_1){
      std::cout<<"ERROR: UNEQUAL VALUES ON TEST #"<<test<<" for add-1!"<<std::endl;
      std::cout<<"Reference      = "<<ref_val                    <<std::endl;
      std::cout<<"Current        = "<<my_val_1                   <<std::endl;
      std::cout<<"Reference bits = "<<binrep<FloatType>(ref_val) <<std::endl;
      std::cout<<"Current   bits = "<<binrep<FloatType>(my_val_1)<<std::endl;
      throw std::runtime_error("Values were not equal!");
    }

    time_deterministic_many.start();
    bitwise_deterministic_summation_many<FloatType>(rfa_result_d, rfa_partial_d, floats_d);
    const auto my_val_many = rfa_result_h->conv();
    time_deterministic_many.stop();
    if(ref_val!=my_val_many){
      std::cout<<"ERROR: UNEQUAL VALUES ON TEST #"<<test<<" for add-many!"<<std::endl;
      std::cout<<"Reference      = "<<ref_val                       <<std::endl;
      std::cout<<"Current        = "<<my_val_many                   <<std::endl;
      std::cout<<"Reference bits = "<<binrep<FloatType>(ref_val)    <<std::endl;
      std::cout<<"Current   bits = "<<binrep<FloatType>(my_val_many)<<std::endl;
      throw std::runtime_error("Values were not equal!");
    }

    time_deterministic_manyc.start();
    bitwise_deterministic_summation_manyc<FloatType>(rfa_result_d, rfa_partial_d, floats_d, 1000);
    const auto my_val_manyc = rfa_result_h->conv();
    time_deterministic_manyc.stop();
    if(ref_val!=my_val_manyc){
      std::cout<<"ERROR: UNEQUAL VALUES ON TEST #"<<test<<" for add-manyc!"<<std::endl;
      std::cout<<"Reference      = "<<ref_val                        <<std::endl;
      std::cout<<"Current        = "<<my_val_manyc                   <<std::endl;
      std::cout<<"Reference bits = "<<binrep<FloatType>(ref_val)     <<std::endl;
      std::cout<<"Current   bits = "<<binrep<FloatType>(my_val_manyc)<<std::endl;
      throw std::runtime_error("Values were not equal!");
    }

    time_kahan.start();
    kahan_summation<kahan<SimpleAccumType>, FloatType>(kahan_result_d, kahan_partial_d, floats_d);
    const auto kahan_sum = kahan_result_h->sum + kahan_result_h->c;
    if (std::abs(kahan_sum - kahan_ldsum) > kahan_max_diff) kahan_max_diff = std::abs(kahan_sum - kahan_ldsum);
    kahan_sums[kahan_sum]++;
    time_kahan.stop();

    time_simple.start();
    simple_summation<SimpleAccumType, FloatType>(result_d, partial_d, floats_d);
    const auto simple_sum = *result_h;
    if (std::abs(simple_sum - kahan_ldsum) > simple_max_diff) simple_max_diff = std::abs(simple_sum - kahan_ldsum);
    simple_sums[simple_sum]++;
    time_simple.stop();
  }

  size_t bytes = floats.size() * sizeof(FloatType);

  std::cout<<"Average deterministic sum 1ata bandwidth  = "<<1e-9*bytes/(time_deterministic_1.total/TESTS)<<" GB/s"<<std::endl;
  std::cout<<"Average deterministic sum many bandwidth  = "<<1e-9*bytes/(time_deterministic_many.total/TESTS)<<" GB/s"<<std::endl;
  std::cout<<"Average deterministic sum manyc bandwidth = "<<1e-9*bytes/(time_deterministic_manyc.total/TESTS)<<" GB/s"<<std::endl;
  std::cout<<"Average simple summation bandwidth        = "<<1e-9*bytes/(time_simple.total/TESTS)<<" GB/s"<<std::endl;
  std::cout<<"Average Kahan summation bandwidth         = "<<1e-9*bytes/(time_kahan.total/TESTS)<<" GB/s"<<std::endl;
  std::cout<<"Average deterministic sum 1ata time       = "<<(time_deterministic_1.total/TESTS)<<std::endl;
  std::cout<<"Average deterministic sum many time       = "<<(time_deterministic_many.total/TESTS)<<std::endl;
  std::cout<<"Average deterministic sum manyc time      = "<<(time_deterministic_manyc.total/TESTS)<<std::endl;
  std::cout<<"Average simple summation time             = "<<(time_simple.total/TESTS)<<std::endl;
  std::cout<<"Average Kahan summation time              = "<<(time_kahan.total/TESTS)<<std::endl;
  std::cout<<"Ratio Deterministic 1ata to Simple        = "<<(time_deterministic_1.total/time_simple.total)<<std::endl;
  std::cout<<"Ratio Deterministic 1ata to Kahan         = "<<(time_deterministic_1.total/time_kahan.total)<<std::endl;
  std::cout<<"Ratio Deterministic many to Simple        = "<<(time_deterministic_many.total/time_simple.total)<<std::endl;
  std::cout<<"Ratio Deterministic many to Kahan         = "<<(time_deterministic_many.total/time_kahan.total)<<std::endl;
  std::cout<<"Ratio Deterministic manyc to Simple       = "<<(time_deterministic_manyc.total/time_simple.total)<<std::endl;
  std::cout<<"Ratio Deterministic manyc to Kahan        = "<<(time_deterministic_manyc.total/time_kahan.total)<<std::endl;

  std::cout<<"Error bound                          = "<<ReproducibleFloatingAccumulator<FloatType>::error_bound(floats.size(), 1000, ref_val)<<std::endl;

  std::cout<<"Reference value                      = "<<std::fixed<<ref_val<<std::endl;
  std::cout<<"Reference bits                       = "<<binrep<FloatType>(ref_val)<<std::endl;

  std::cout<<"Kahan long double accumulator value  = "<<kahan_ldsum<<std::endl;
  std::cout<<"Distinct Kahan values                = "<<kahan_sums.size()<<std::endl;
  std::cout<<"Distinct Simple values               = "<<simple_sums.size()<<std::endl;
  std::cout<<"Deterministic deviation              = "<<ref_diff<<std::endl;
  std::cout<<"Kahan max abs deviation              = "<<kahan_max_diff<<std::endl;
  std::cout<<"Simple max abs deviation             = "<<simple_max_diff<<std::endl;

#if 0
  for(const auto &kv: kahan_sums){
    std::cout<<"Kahan sum values (N="<<std::fixed<<kv.second<<") "<<kv.first<<" ("<<binrep<FloatType>(kv.first)<<")"<<std::endl;
  }

  for(const auto &kv: simple_sums){
    std::cout<<"Simple sum values (N="<<std::fixed<<kv.second<<") "<<kv.first<<" ("<<binrep<FloatType>(kv.first)<<")"<<std::endl;
  }
#endif
  std::cout<<std::endl;

  CHECK_CUDA(cudaFree(kahan_partial_d));
  CHECK_CUDA(cudaFreeHost(kahan_result_h));

  CHECK_CUDA(cudaFree(partial_d));
  CHECK_CUDA(cudaFreeHost(result_h));

  CHECK_CUDA(cudaFree(rfa_partial_d));
  CHECK_CUDA(cudaFreeHost(rfa_result_h));

  return ref_val;
}

// Use this to make sure the tests are reproducible
template<class FloatType, class SimpleAccumType>
void PerformTestsOnUniformRandom(const int N, const int TESTS){
  thrust::host_vector<FloatType> input;
  {
    nvtx3::scoped_range r{"setup"};
    std::mt19937 gen(123456789);
    std::uniform_real_distribution<double> distr(-1000, 1000);
    thrust::host_vector<double> floats;
    for (int i=0;i<N;i++) floats.push_back(distr(gen));
    input = {floats.begin(), floats.end()};
    std::cout<<"Input Data                           = Uniform Random"<<std::endl;
  }
  PerformTestsOnData<FloatType, SimpleAccumType>(TESTS, input);
}

// Use this to make sure the tests are reproducible
template<class FloatType, class SimpleAccumType>
void PerformTestsOnSineWaveData(const int N, const int TESTS){
  thrust::host_vector<FloatType> input;
  {
    nvtx3::scoped_range r{"setup"};
    input.reserve(N);
    // Make a sine wave
    for(int i = 0; i < N; i++){
      input.push_back(std::sin(2 * M_PI * (i / static_cast<double>(N) - 0.5)));
    }
    std::cout<<"Input Data                           = Sine Wave"<<std::endl;
  }
  PerformTestsOnData<FloatType, SimpleAccumType>(TESTS, input);
}

int main(){
  const int N = 1'000'000;
  const int TESTS = 100;

  std::cout << "Running CUDA parallel summation tests" << std::endl;
  std::cout << "N = " << N << std::endl;
  std::cout << "TESTS = " << TESTS << std::endl;
  std::cout << "grid size = " << grid_size << std::endl;
  std::cout << "block size = " << block_size << std::endl;
  std::cout << std::endl;

  {
    nvtx3::scoped_range r{"uniform float"};
    PerformTestsOnUniformRandom<float, float>(N, TESTS);
  }

  {
    nvtx3::scoped_range r{"uniform double"};
    PerformTestsOnUniformRandom<double, double>(N, TESTS);
  }

  {
    nvtx3::scoped_range r{"sine double"};
    PerformTestsOnSineWaveData<float, float>(N, TESTS);
  }

  {
    nvtx3::scoped_range r{"sine float"};
    PerformTestsOnSineWaveData<double, double>(N, TESTS);
  }

  return 0;
}
