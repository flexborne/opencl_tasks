__kernel void add_vectors(__global const float *a, 
	__global const float *b,
	__global float *c) 
{
  unsigned gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];

// https://www.intel.com/content/www/us/en/docs/programmable/683176/18-1/kernel-vectorization-opencl-standard.html
//  c[gid * 4 + 0] = a[gid * 4 + 0] + b[gid * 4 + 0];
//  c[gid * 4 + 1] = a[gid * 4 + 1] + b[gid * 4 + 1];
//  c[gid * 4 + 2] = a[gid * 4 + 2] + b[gid * 4 + 2];
//  c[gid * 4 + 3] = a[gid * 4 + 3] + b[gid * 4 + 3];
// such a bad performance in result of this
}

#define TS 16 


__kernel void matmul(const __global float* A,
                      const __global float* B,
                      __global float* C, 
					  const unsigned M, const unsigned N, const unsigned K) 
{
  const unsigned col = get_local_id(1); 
  const unsigned row = get_local_id(0);
  const unsigned global_row = TS*get_group_id(0) + row; 
  const unsigned global_col = TS*get_group_id(1) + col; 
  
  __local float A_sub[TS][TS];
  __local float B_sub[TS][TS];
  
  float acc = 0.f;
  
  const unsigned tiles = K/TS;
  for (unsigned t = 0; t < tiles; ++t) {
  	const int tiled_row = TS*t + row;
  	const int tiled_col = TS*t + col;
  	A_sub[col][row] = A[global_col*K + tiled_row];
  	B_sub[col][row] = B[tiled_col*M + global_row];
  
  	barrier(CLK_LOCAL_MEM_FENCE);
  
  	for (int k = 0; k < TS; ++k) {
  	  acc += B_sub[k][row] * A_sub[col][k];
  	}
  	barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  C[global_col*M + global_row] = acc;
}


__kernel void reduce(__global const float *input, __global float *output, const unsigned n) {
  __local float local_sum[256];
  
  const int global_id = get_global_id(0);
  const int local_id = get_local_id(0);
  const int group_size = get_local_size(0);
  
  if (global_id < n) {
    local_sum[local_id] = input[global_id];
  } else {
    local_sum[local_id] = 0.0f; 
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for (int offset = group_size / 2; offset > 0; offset >>= 1) {
  	if (local_id < offset) {
  		local_sum[local_id] += local_sum[local_id + offset];
  	}
  	barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if (local_id == 0) {
    output[get_group_id(0)] = local_sum[0];
  }
}

#define VECTOR_SORT(input, dir) \
 comp = abs(input > shuffle(input, mask2)) ^ dir; \
 input = shuffle(input, comp * 2 + add2); \
 comp = abs(input > shuffle(input, mask1)) ^ dir; \
 input = shuffle(input, comp + add1); \

 #define VECTOR_SORT_NO_ABS(input, dir) \
 comp = (input > shuffle(input, mask2)) ^ dir; \
 input = shuffle(input, as_uint4(comp * 2 + add2)); \
 comp = (input > shuffle(input, mask1)) ^ dir; \
 input = shuffle(input, as_uint4(comp + add1)); \

#define VECTOR_SWAP(in1, in2, dir) \
 input1 = in1; input2 = in2; \
 comp = (abs(input1 > input2) ^ dir) * 4 + add3; \
 in1 = shuffle2(input1, input2, comp); \
 in2 = shuffle2(input2, input1, comp); \

#define VECTOR_SWAP_NO_ABS(in1, in2, dir) \
 input1 = in1; input2 = in2; \
 comp = ((input1 > input2) ^ dir) * 4 + add3; \
 in1 = shuffle2(input1, input2, as_uint4(comp)); \
 in2 = shuffle2(input2, input1, as_uint4(comp)); \

__kernel void bsort_init(__global float4 *g_data,
  __local float4 *l_data) {
  float4 input1, input2, temp;
  uint4 comp, swap, mask1, mask2, add1, add2, add3;
  uint id, dir, global_start, size, stride;
  mask1 = (uint4)(1, 0, 3, 2);
  swap = (uint4)(0, 0, 1, 1);
  add1 = (uint4)(0, 0, 2, 2);
  mask2 = (uint4)(2, 3, 0, 1);
  add2 = (uint4)(0, 1, 0, 1);
  add3 = (uint4)(0, 1, 2, 3);
  id = get_local_id(0) * 2;
  global_start = get_group_id(0) * get_local_size(0) * 2 + id;

  input1 = g_data[global_start];
  input2 = g_data[global_start + 1];

  comp = abs(input1 > shuffle(input1, mask1));
  input1 = shuffle(input1, comp ^ swap + add1);
  comp = abs(input1 > shuffle(input1, mask2));
  input1 = shuffle(input1, comp * 2 + add2);
  comp = abs(input1 > shuffle(input1, mask1));
  input1 = shuffle(input1, comp + add1);
  comp = abs(input2 < shuffle(input2, mask1));
  input2 = shuffle(input2, comp ^ swap + add1);
  comp = abs(input2 < shuffle(input2, mask2));
  input2 = shuffle(input2, comp * 2 + add2);
  comp = abs(input2 < shuffle(input2, mask1));
  input2 = shuffle(input2, comp + add1);

  dir = get_local_id(0) % 2;
  temp = input1;
  comp = (abs(input1 > input2) ^ dir) * 4 + add3;
  input1 = shuffle2(input1, input2, comp);
  input2 = shuffle2(input2, temp, comp);

  VECTOR_SORT(input1, dir);
  VECTOR_SORT(input2, dir);

  l_data[id] = input1;
  l_data[id + 1] = input2;

  for (size = 2; size < get_local_size(0); size <<= 1) {
    dir = get_local_id(0)/size & 1;
    for (stride = size; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    id = get_local_id(0) * 2;
    input1 = l_data[id]; input2 = l_data[id+1];
    temp = input1;
    comp = (abs(input1 > input2) ^ dir) * 4 + add3;
    input1 = shuffle2(input1, input2, comp);
    input2 = shuffle2(input2, temp, comp);
    VECTOR_SORT(input1, dir);
    VECTOR_SORT(input2, dir);
    l_data[id] = input1;
    l_data[id+1] = input2;
  }
  dir = get_group_id(0) % 2;
  for (stride = get_local_size(0); stride > 1; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    id = get_local_id(0) + (get_local_id(0)/stride)*stride;
    VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  id = get_local_id(0) * 2;
  input1 = l_data[id]; input2 = l_data[id+1];
  temp = input1;
  comp = (abs(input1 > input2) ^ dir) * 4 + add3;
  input1 = shuffle2(input1, input2, comp);
  input2 = shuffle2(input2, temp, comp);
  VECTOR_SORT(input1, dir);
  VECTOR_SORT(input2, dir);
  g_data[global_start] = input1;
  g_data[global_start+1] = input2;
} 

__kernel void bsort_stage_0(__global float4 *g_data, __local float4 *l_data, 
                            uint high_stage) {
   int dir;
   uint id, global_start, stride;
   float4 input1, input2, temp;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   id = get_local_id(0);
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   input1 = g_data[global_start];
   input2 = g_data[global_start + get_local_size(0)];
   comp = (input1 < input2 ^ dir) * 4 + add3;
   l_data[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

   for (stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    id = get_local_id(0) + (get_local_id(0)/stride)*stride;
    VECTOR_SWAP_NO_ABS(l_data[id], l_data[id + stride], dir)
  }
  barrier(CLK_LOCAL_MEM_FENCE);

   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   temp = input1;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   VECTOR_SORT_NO_ABS(input1, dir);
   VECTOR_SORT_NO_ABS(input2, dir);

   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;
}

__kernel void bsort_stage_n(__global float4 *g_data, __local float4 *l_data, 
                            uint stage, uint high_stage) {
  int dir;
  float4 input1, input2;
  int4 comp, add;
  uint global_start, global_offset;

  add = (int4)(4, 5, 6, 7);

  dir = (get_group_id(0)/high_stage & 1) * -1;
  global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                  get_local_size(0) + get_local_id(0);
  global_offset = stage * get_local_size(0);

  /* Perform swap */
  input1 = g_data[global_start];
  input2 = g_data[global_start + global_offset];
  comp = (input1 < input2 ^ dir) * 4 + add;
  g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
  g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

__kernel void bsort_merge(__global float4 *g_data, __local float4 *l_data, uint stage, int dir) {

   float4 input1, input2;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   /* Perform swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + global_offset];
   comp = ((input1 < input2) ^ dir) * 4 + add;
   g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

__kernel void bsort_merge_last(__global float4 *g_data, __local float4 *l_data, int dir) {

   uint id, global_start, stride;
   float4 input1, input2, temp;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   id = get_local_id(0);
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   input1 = g_data[global_start];
   input2 = g_data[global_start + get_local_size(0)];
   comp = (input1 < input2 ^ dir) * 4 + add3;

   l_data[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

   /* Perform bitonic merge */
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP_NO_ABS(l_data[id], l_data[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   temp = input1;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   VECTOR_SORT_NO_ABS(input1, dir);
   VECTOR_SORT_NO_ABS(input2, dir);

   /* Store the result to global memory */
   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;

}