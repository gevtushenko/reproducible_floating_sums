#pragma once

namespace detail {
  template <class T> struct get_vector_type { };
  template <> struct get_vector_type<double> { using type = double2; };
  template <> struct get_vector_type<float> { using type = float4; };
}

template <class T> using vector_t = typename detail::get_vector_type<T>::type;

template <class T> auto vector_size() { return sizeof(vector_t<T>) / sizeof(T); }

template <class T> auto abs_max(const T&);

template <> __host__ __device__ auto abs_max(const float4 &x) { return fmax( fmaxf(fabs(x.x), fabs(x.y)), fmax(fabs(x.z), fabs(x.w)) ); }

template <> __host__ __device__ auto abs_max(const double2 &x) { return fmax(fabs(x.x), fabs(x.y)); }

template <> __host__ __device__ auto abs_max(const float2 &x) { return fmax(fabs(x.x), fabs(x.y)); }

template <class T> auto reduce(const T &);
template <> __host__ __device__ auto reduce(const float &x) { return x; }
template <> __host__ __device__ auto reduce(const double &x) { return x; }
template <> __host__ __device__ auto reduce(const float4 &x) { return x.x + x.y + x.z + x.w; }
template <> __host__ __device__ auto reduce(const float2 &x) { return x.x + x.y; }
template <> __host__ __device__ auto reduce(const double2 &x) { return x.x + x.y; }
