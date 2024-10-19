#include "tasks.h"
#include <CL/cl.h>

#include "common.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <ranges>
#include <span>
#include <numeric>


namespace {

class stop_watch {
 private:
  using clock = std::chrono::high_resolution_clock;

 private:
  std::chrono::time_point<clock> start_time;

 public:
  void start() noexcept { start_time = clock::now(); }

  auto duration() const noexcept { return (clock::now() - start_time).count(); }
};

template <class T>
void print(std::span<const T> range, std::string_view msg) {
  std::cout << msg << '\n';
  std::ranges::copy(range, std::ostream_iterator<T>(std::cout, " "));
  std::cout << '\n';
}

float generate_random_number(int f_min, int f_max) {
  assert(f_max > f_min);
  double f = rand() / static_cast<double>(RAND_MAX);
  return f_min + f * (f_max - f_min);
}

template <class T>
void init_arr(std::span<T> arr, int f_min = -10, int f_max = 10) {
  for (size_t i = 0; i < arr.size(); ++i) {
    arr[i] = generate_random_number(f_min, f_max);
  }
}

template <class T>
constexpr bool equal(T a, T b, T eps = 1e-2) noexcept {
  return std::abs(a - b) < eps;
}

template <class T>
inline bool equal(const T* lhs, const T* rhs, size_t n) noexcept {
  for (size_t i = 0; i < n; ++i) {
    if (!equal<T>(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

double event_duration(cl_event event) 
{
  cl_ulong start_time, end_time;
  auto ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
  assert(ret == CL_SUCCESS);

  ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
  assert(ret == CL_SUCCESS);

  clReleaseEvent(event);

  return end_time - start_time;
}

}  // namespace

namespace t1 {

constexpr inline auto N = 100000uz;
using type = float;
using array_t = std::array<type, N>;

auto init_arr() noexcept {
  array_t arr;
  ::init_arr<type>(arr, -1, 1);
  return arr;
};

}  // namespace t1

void task1(bool print_arrays) {
  using namespace t1;

  std::cout << std::format("TASK 1, N = {}\n", N);

  const array_t a = init_arr();
  const array_t b = init_arr();

  array_t res_cl;
  
  constexpr static auto bytes = N * sizeof(type);

  cl_int ret;
  cl_mem a_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &ret);
  assert(ret == CL_SUCCESS);
	cl_mem b_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &ret);
  assert(ret == CL_SUCCESS);
	cl_mem c_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &ret);
  assert(ret == CL_SUCCESS);

	ret = clEnqueueWriteBuffer(queue, a_cl, CL_TRUE, 0, bytes, a.data(), 0, NULL, NULL);
  assert(ret == CL_SUCCESS);

	ret = clEnqueueWriteBuffer(queue, b_cl, CL_TRUE, 0, bytes, b.data(), 0, NULL, NULL);
  assert(ret == CL_SUCCESS);

  cl_kernel kernel = clCreateKernel(program, "add_vectors", &ret);
  assert(ret == CL_SUCCESS);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_cl);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_cl);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_cl);

  constexpr size_t global_ws = N;  
  constexpr size_t local_ws = 50; 
  static_assert(global_ws % local_ws == 0);

  cl_event kernel_event;
  ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, &kernel_event);
  assert(ret == CL_SUCCESS);
  clWaitForEvents(1, &kernel_event);

  const auto duration_cl = event_duration(kernel_event);

  ret = clEnqueueReadBuffer(queue, c_cl, CL_TRUE, 0, bytes, res_cl.data(), 0, NULL, NULL);
  assert(ret == CL_SUCCESS);

  clReleaseMemObject(a_cl);
  clReleaseMemObject(b_cl);
  clReleaseMemObject(c_cl);
  clReleaseKernel(kernel);

  array_t res_loop;
  stop_watch sw;
  sw.start();
  sum_loop(a.data(), b.data(), res_loop.data(), N);
  const auto duration_loop = sw.duration();

  assert(equal(res_loop.data(), res_cl.data(), N));

  if (print_arrays) {
    print<type>(res_loop, "Loop res:");
    print<type>(res_cl, "CL res:");
  }

  print_durations(duration_loop, duration_cl);
}

namespace t2 {
constexpr unsigned M = 1024;
constexpr unsigned N = 1024;
constexpr unsigned K = 1024;

using type = float;

auto init_arr(size_t n) noexcept {
  std::vector<type> arr(n);
  ::init_arr<type>(arr);
  return arr;
};

}  // namespace t2

void task2(bool print_arrays) {
  using namespace t2;

  std::cout << std::format("TASK 2, M = {}, N = {}, K = {}\n", M, N, K);

  const auto a = init_arr(M*N);
  const auto b = init_arr(N*K);
  auto res_cpu = std::vector<type>(M*K, 0);
  auto res_cl = std::vector<type>(M*K, 0);

  
  cl_int ret;
  cl_mem a_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, a.size()*sizeof(type), NULL, &ret);
  assert(ret == CL_SUCCESS);
	cl_mem b_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, b.size()*sizeof(type), NULL, &ret);
  assert(ret == CL_SUCCESS);
	cl_mem c_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, res_cl.size()*sizeof(type), NULL, &ret);
  assert(ret == CL_SUCCESS);

	ret = clEnqueueWriteBuffer(queue, a_cl, CL_TRUE, 0, a.size()*sizeof(type), a.data(), 0, NULL, NULL);
  assert(ret == CL_SUCCESS);

	ret = clEnqueueWriteBuffer(queue, b_cl, CL_TRUE, 0, b.size()*sizeof(type), b.data(), 0, NULL, NULL);
  assert(ret == CL_SUCCESS);

  cl_kernel kernel = clCreateKernel(program, "matmul", &ret);
  assert(ret == CL_SUCCESS);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_cl);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_cl);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_cl);
  ret = clSetKernelArg(kernel, 3, sizeof(unsigned), (void*)&M);
  ret = clSetKernelArg(kernel, 4, sizeof(unsigned), (void*)&N);
  ret = clSetKernelArg(kernel, 5, sizeof(unsigned), (void*)&K);

  constexpr size_t global_ws[2] = {M, N};  
  constexpr size_t local_ws[2] = {16, 16}; 
  static_assert(global_ws[0] % local_ws[0] == 0);
  static_assert(global_ws[1] % local_ws[1] == 0);

  cl_event kernel_event;
  ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_ws, local_ws, 0,
                               NULL, &kernel_event);
  assert(ret == CL_SUCCESS);
  clWaitForEvents(1, &kernel_event);

  const auto duration_cl = event_duration(kernel_event);

  ret = clEnqueueReadBuffer(queue, c_cl, CL_TRUE, 0, res_cl.size()*sizeof(type), res_cl.data(), 0, NULL, NULL);
  assert(ret == CL_SUCCESS);

  clReleaseMemObject(a_cl);
  clReleaseMemObject(b_cl);
  clReleaseMemObject(c_cl);
  clReleaseKernel(kernel);

  stop_watch sw;
  sw.start();
  matmul_loop(a.data(), b.data(), res_cpu.data(), M, N, K);
  const auto duration_loop = sw.duration();

  assert(equal(res_cpu.data(), res_cl.data(), N));

  if (print_arrays) {
    print<type>(res_cpu, "Loop res:");
    print<type>(res_cl, "CL res:");
  }

  print_durations(duration_loop, duration_cl);
}
namespace t3 {

constexpr unsigned GROUP_SZ = 256;
constexpr unsigned N = GROUP_SZ*GROUP_SZ*GROUP_SZ;
using type = float;

auto init_arr() noexcept {
  std::vector<type> arr(N);
  ::init_arr<type>(arr, -1, 1);
  return arr;
};

}  // namespace t3

void task3() {
  using namespace t3;

  std::cout << std::format("TASK 3, N = {}\n", N);

  const auto a = init_arr();
  stop_watch sw;

  cl_int ret;
  cl_mem a_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(type), NULL, &ret);
  assert(ret == CL_SUCCESS);

  constexpr size_t n_outputs = N / GROUP_SZ;
  cl_mem output_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_outputs * sizeof(type), NULL, NULL);
  assert(ret == CL_SUCCESS);

	ret = clEnqueueWriteBuffer(queue, a_cl, CL_TRUE, 0, N * sizeof(type), a.data(), 0, NULL, NULL);
  assert(ret == CL_SUCCESS);

  cl_kernel kernel = clCreateKernel(program, "reduce", &ret);
  assert(ret == CL_SUCCESS);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_cl);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output_cl);
  ret = clSetKernelArg(kernel, 2, sizeof(unsigned), (void*)&N);

  constexpr size_t global_ws = N;  
  constexpr size_t local_ws = GROUP_SZ; 
  static_assert(global_ws % local_ws == 0);

  cl_event reduce_event;
  ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, &reduce_event);
  assert(ret == CL_SUCCESS);
  clWaitForEvents(1, &reduce_event);
  auto duration_cl = event_duration(reduce_event);

  const auto SECOND_ROUND_SZ = N/GROUP_SZ;
  cl_event reduce_event2;
  a_cl = output_cl;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_cl);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output_cl);
  ret |= clSetKernelArg(kernel, 2, sizeof(unsigned), (void*)&SECOND_ROUND_SZ);

  ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, &reduce_event2);
  assert(ret == CL_SUCCESS);
  clWaitForEvents(1, &reduce_event2);
  duration_cl += event_duration(reduce_event2);

  std::array<type, GROUP_SZ> outputs;
  ret = clEnqueueReadBuffer(queue, output_cl, CL_TRUE, 0, GROUP_SZ*sizeof(type), outputs.data(), 0, NULL, NULL);
  assert(ret == CL_SUCCESS);
  // actually CAN do one more time, too lazy to write lambda though and generalize
  sw.start();
  const auto res_cl = std::reduce(outputs.begin(), outputs.end(), 0.f);
  duration_cl += sw.duration();

  clReleaseMemObject(a_cl);
  clReleaseMemObject(output_cl);
  clReleaseKernel(kernel);

  sw.start();
  const auto res_loop = std::reduce(a.begin(), a.end(), 0.f);
  const auto duration_loop = sw.duration();

  assert(equal(res_loop, res_cl, 1.f));

  std::cout << "Loop res: " << res_loop << "\nRes CL: " << res_cl << "\n";

  print_durations(duration_loop, duration_cl);
}

namespace t4 {

constexpr unsigned N = 256*256*256;
using type = float;

auto init_arr() noexcept {
  std::vector<type> arr(N);
  ::init_arr<type>(arr, -1, 1);
  return arr;
};

}  // namespace t4

void task4(bool print_arrays) {
  using namespace t4;

  std::cout << std::format("TASK 4, N = {}\n", N);
  std::vector<cl_event> events;
  events.reserve(1'000);

  auto arr = init_arr();
  std::vector<type> res_cl(N);
  stop_watch sw;

  cl_int ret;
  cl_mem arr_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(type), NULL, &ret);
  assert(ret == CL_SUCCESS);

	ret = clEnqueueWriteBuffer(queue, arr_cl, CL_TRUE, 0, N * sizeof(type), arr.data(), 0, NULL, NULL);
  assert(ret == CL_SUCCESS);

  auto kernel_init = clCreateKernel(program, "bsort_init", &ret);
  assert(ret == CL_SUCCESS);
  auto kernel_stage_0 = clCreateKernel(program, "bsort_stage_0", &ret);
  assert(ret == CL_SUCCESS);
  auto kernel_stage_n = clCreateKernel(program, "bsort_stage_n", &ret);
  assert(ret == CL_SUCCESS);
  auto kernel_merge = clCreateKernel(program, "bsort_merge", &ret);
  assert(ret == CL_SUCCESS);
  auto kernel_merge_last = clCreateKernel(program, "bsort_merge_last", &ret);
  assert(ret == CL_SUCCESS);

  constexpr size_t global_ws = N/8;  
  size_t local_ws;

  ret = clGetKernelWorkGroupInfo(kernel_init, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(local_ws), &local_ws, NULL);
  assert(ret == CL_SUCCESS);
  assert(global_ws % local_ws == 0);

  ret = clSetKernelArg(kernel_init, 0, sizeof(cl_mem), &arr_cl);
  ret |= clSetKernelArg(kernel_stage_0, 0, sizeof(cl_mem), &arr_cl);
  ret |= clSetKernelArg(kernel_stage_n, 0, sizeof(cl_mem), &arr_cl);
  ret |= clSetKernelArg(kernel_merge, 0, sizeof(cl_mem), &arr_cl);
  ret |= clSetKernelArg(kernel_merge_last, 0, sizeof(cl_mem), &arr_cl);
  assert(ret == CL_SUCCESS);

  ret = clSetKernelArg(kernel_init, 1, 8 * local_ws * sizeof(type), NULL);
  ret |= clSetKernelArg(kernel_stage_0, 1, 8 * local_ws * sizeof(type), NULL);
  ret |= clSetKernelArg(kernel_stage_n, 1, 8 * local_ws * sizeof(type), NULL);
  ret |= clSetKernelArg(kernel_merge, 1, 8 * local_ws * sizeof(type), NULL);
  ret |= clSetKernelArg(kernel_merge_last, 1, 8 * local_ws * sizeof(type), NULL);
  assert(ret == CL_SUCCESS);

  auto& init_event = events.emplace_back();
  ret = clEnqueueNDRangeKernel(queue, kernel_init, 1, NULL, &global_ws,
                               &local_ws, 0, NULL, &init_event);
  assert(ret == CL_SUCCESS);

  const auto num_stages = global_ws / local_ws;
  for (auto high_stage = 2; high_stage < num_stages; high_stage <<= 1) {
    ret = clSetKernelArg(kernel_stage_0, 2, sizeof(int), &high_stage);
    ret |= clSetKernelArg(kernel_stage_n, 3, sizeof(int), &high_stage);
    assert(ret == CL_SUCCESS);

    for (auto stage = high_stage; stage > 1; stage >>= 1) {

      ret = clSetKernelArg(kernel_stage_n, 2, sizeof(int), &stage);
      assert(ret == CL_SUCCESS);

      auto& event = events.emplace_back();
      ret = clEnqueueNDRangeKernel(queue, kernel_stage_n, 1, NULL, &global_ws,
                                   &local_ws, 0, NULL, &event);
      assert(ret == CL_SUCCESS);
    }

    auto& event = events.emplace_back();
    ret = clEnqueueNDRangeKernel(queue, kernel_stage_0, 1, NULL, &global_ws,
                                 &local_ws, 0, NULL, &event);
    assert(ret == CL_SUCCESS);
  }

  auto direction = 0;
  ret = clSetKernelArg(kernel_merge, 3, sizeof(int), &direction);
  ret |= clSetKernelArg(kernel_merge_last, 2, sizeof(int), &direction);
  assert(ret == CL_SUCCESS);

  for (auto stage = num_stages; stage > 1; stage >>= 1) {
    ret = clSetKernelArg(kernel_merge, 2, sizeof(int), &stage);
    assert(ret == CL_SUCCESS);

    auto& event = events.emplace_back();
    ret = clEnqueueNDRangeKernel(queue, kernel_merge, 1, NULL, &global_ws,
                                 &local_ws, 0, NULL, &event);
    assert(ret == CL_SUCCESS);
  }
  auto& merge_last_event = events.emplace_back();
  ret = clEnqueueNDRangeKernel(queue, kernel_merge_last, 1, NULL, &global_ws,
                               &local_ws, 0, NULL, &merge_last_event);
  assert(ret == CL_SUCCESS);

  clWaitForEvents(events.size(), events.data());
  double duration_cl = 0;
  for (auto& event : events) {
    duration_cl += event_duration(event);
  }

  ret = clEnqueueReadBuffer(queue, arr_cl, CL_TRUE, 0, N*sizeof(type), res_cl.data(),
                            0, NULL, NULL);
  assert(ret == CL_SUCCESS);

  clReleaseMemObject(arr_cl);
  clReleaseKernel(kernel_init);
  clReleaseKernel(kernel_stage_0);
  clReleaseKernel(kernel_stage_n);
  clReleaseKernel(kernel_merge);
  clReleaseKernel(kernel_merge_last);

  sw.start();
  std::sort(arr.begin(), arr.end());
  const auto duration_loop = sw.duration();

  assert(equal<type>(arr.data(), res_cl.data(), N));

  if (print_arrays) {
    print<type>(arr, "Loop res:");
    print<type>(res_cl, "CL res:");
  }

  print_durations(duration_loop, duration_cl);
}
