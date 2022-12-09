#include <cub/cub.cuh>

constexpr int grid_size = 160;    // blocks per grid
constexpr int block_size = 512; // threads per block

#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ unsigned int count = 0;

// we can make this much better by first computing the block wide max
// and then summing
template <int block_size, class T>
__device__ auto block_sum(T value)
{
  // Specialize BlockReduce for a 1D block of 128 threads of type int
  using BlockReduce = cub::BlockReduce<T, block_size>;
  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;
  // Compute the block-wide sum for thread0
  return BlockReduce(temp_storage).Sum(value);
}

template <class T> constexpr
std::enable_if_t<std::is_same_v<T, ReproducibleFloatingAccumulator<typename T::ftype, T::FOLD>>, bool> is_rfa()
{ return true; }

template <class T> constexpr
std::enable_if_t<!std::is_same_v<T, ReproducibleFloatingAccumulator<typename T::ftype, T::FOLD>>, bool> is_rfa()
{ return false; }

template <int block_size, class T, class index_t>
__device__ void reduce(T *result, T *partial, T &value, index_t tid)
{
  if constexpr (grid_size == 1) {

    auto aggregate = block_sum<block_size>(value);
    if (tid == 0) *result = aggregate;

  } else {

    __shared__ bool is_last_block_done;
    auto aggregate = block_sum<block_size>(value);

    if (threadIdx.x == 0) {
      partial[blockIdx.x] = aggregate; // non-coalesced write
      __threadfence(); // flush result

      // increment global block counter
      auto value = atomicInc(&count, gridDim.x);
      is_last_block_done = (value == gridDim.x - 1);
    }

    __syncthreads();

    // finish reduction if last block
    if (is_last_block_done) {
      auto i = threadIdx.x;
      T sum = { };
      while (i < gridDim.x) {
        sum += const_cast<T&>(static_cast<volatile T*>(partial)[i]); // non-coalesced read
        i += blockDim.x;
      }

      auto aggregate = block_sum<block_size>(sum);
      if (threadIdx.x == 0) *result = aggregate;

      count = 0; // reset counter
    }

  }
}

// Kahan summation (buggy in parallel where we seem to lose some
// precision)
template <class AccumType>
struct kahan {
  AccumType sum = 0.0;
  AccumType c = 0.0;

  template <class FloatType>
  __host__ __device__ void operator+=(const FloatType &x) {
    const auto y = x - c;
    const auto t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

#ifdef __CUDACC__
  __host__ __device__ void operator+=(const double2 &x) {
    operator+=(x.x);
    operator+=(x.y);
  }

  __host__ __device__ void operator+=(const float2 &x) {
    operator+=(x.x);
    operator+=(x.y);
  }

  __host__ __device__ void operator+=(const float4 &x) {
    operator+=(x.x);
    operator+=(x.y);
    operator+=(x.z);
    operator+=(x.w);
  }
#endif

  // adding multiple Kahan accumulators lose precision?
  __host__ __device__ void operator+=(kahan<AccumType> x) {
#if 0  // http://faculty.washington.edu/rjl/icerm2012/Lightning/Robey.pdf
    //auto new_c = x.sum - (x.c + c);
    //auto new_sum = sum + new_c;
    //c = (new_sum - sum) - new_c;
    //sum = new_sum;
#else
    // when adding two Kahan accumulators use Neumaier form
    if (std::abs(sum) > std::abs(x.sum)) {
      operator+=(x.c);
      operator+=(x.sum);
    } else {
      x += c;
      x += sum;
      *this = x;
    }
#endif
  }
};


// Add two Kahan accumulators together (used by CUB)
template <class AccumType>
__host__ __device__ auto operator+(const kahan<AccumType> &lhs, const kahan<AccumType> &rhs)
{
  kahan<AccumType> rtn = lhs;
  rtn += rhs;
  return rtn;
}

// Add two RFA together (used by CUB)
template <class Accumulator>
__host__ __device__
std::enable_if_t<std::is_same_v<Accumulator, ReproducibleFloatingAccumulator<typename Accumulator::ftype, Accumulator::FOLD>>, Accumulator>
operator+(const Accumulator &lhs, const Accumulator &rhs)
{
  Accumulator rtn = lhs;
  rtn += rhs;
  return rtn;
}
