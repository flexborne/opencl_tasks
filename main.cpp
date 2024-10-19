#include "tasks.h"
#include "common.h"

#include <chrono>
#include <iostream>
#include <cassert>

#include <CL/cl.h>

cl_context context;
cl_command_queue queue;
cl_program program;
cl_device_id device_id;

int main(int argc, char** argv) {
  std::srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  auto* kernelFile = fopen("../../common.cl", "r");

	if (!kernelFile) {
    assert(false);
		std::cerr << "No file named common.cl was found\n";
		exit(-1);
	}

  std::array<char, 100'000> buff;
  char* kernelSource = buff.data();
  const size_t kernelSize = fread(kernelSource, 1, 100'000, kernelFile);
	fclose(kernelFile);

  cl_platform_id platformId = NULL;
	device_id = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;

	auto ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
  assert(ret == CL_SUCCESS);
  
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &retNumDevices);
  assert(ret == CL_SUCCESS);

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL,  &ret);
  assert(ret == CL_SUCCESS);

  queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
  assert(ret == CL_SUCCESS);

  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, &kernelSize, &ret);	
  assert(ret == CL_SUCCESS);

  ret = clBuildProgram(program, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
  if (ret != CL_SUCCESS) {
    printf("clBuildProgram returned: %d\n", ret);

    // If there was an error during build, get the build log
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    
    // Allocate memory for the log
    char *log = (char *)malloc(log_size);
    
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("Error in kernel: %s\n", log);
    
    free(log);
}
  assert(ret == CL_SUCCESS);


  //task1(false);

  //task2(false);
  //task3();
  task4(false);

	ret = clFlush(queue);
	ret = clFinish(queue);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
