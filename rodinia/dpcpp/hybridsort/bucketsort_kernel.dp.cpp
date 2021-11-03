#ifndef _BUCKETSORT_KERNEL_H_
#define _BUCKETSORT_KERNEL_H_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

#define BUCKET_WARP_LOG_SIZE	5
#define BUCKET_WARP_N			1
#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N			(BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY		(DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND				128

/*
DPCT1059:25: SYCL only supports 4-channel image format. Adjust the code.
*/
//dpct::image_wrapper<float, 1> texPivot;
dpct::image_wrapper<cl::sycl::float4, 1> texPivot;

int addOffset(volatile unsigned int *s_offset, unsigned int data, unsigned int threadTag){
    unsigned int count;

    do{
        count = s_offset[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_offset[data] = count;
    }while(s_offset[data] != count);

	return (count & 0x07FFFFFFU) - 1;
}

void
bucketcount( float *input, int *indice, unsigned int *d_prefixoffsets, int size,
             sycl::nd_item<3> item_ct1, volatile unsigned int *s_offset,
             //dpct::image_accessor_ext<float, 1> texPivot)
             dpct::image_accessor_ext<cl::sycl::float4, 1> texPivot)
{

    const unsigned int threadTag = item_ct1.get_local_id(2)
                                   << (32 - BUCKET_WARP_LOG_SIZE);
    const int warpBase =
        (item_ct1.get_local_id(2) >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads =
        item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);
        for (int i = item_ct1.get_local_id(2); i < BUCKET_BLOCK_MEMORY;
             i += item_ct1.get_local_range().get(2))
                s_offset[i] = 0;

        item_ct1.barrier();

        for (int tid =
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
             tid < size; tid += numThreads) {
                float elem = input[tid]; 

		int idx  = DIVISIONS/2 - 1; 
		int jump = DIVISIONS/4;
                cl::sycl::float4 piv = texPivot.read(idx); // s_pivotpoints[idx];

                while(jump >= 1){
			idx = (elem < piv[0]) ? (idx - jump) : (idx + jump);
                        piv = texPivot.read(idx); // s_pivotpoints[idx];
                        jump /= 2; 
		}
		idx = (elem < piv[0]) ? idx : (idx + 1); 

		indice[tid] = (addOffset(s_offset + warpBase, idx, threadTag) << LOG_DIVISIONS) + idx;  //atomicInc(&offsets[idx], size + 1);
	}

        item_ct1.barrier();

        int prefixBase = item_ct1.get_group(2) * BUCKET_BLOCK_MEMORY;

        for (int i = item_ct1.get_local_id(2); i < BUCKET_BLOCK_MEMORY;
             i += item_ct1.get_local_range().get(2))
                d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU; 
}

void bucketprefixoffset(unsigned int *d_prefixoffsets, unsigned int *d_offsets, int blocks,
                        sycl::nd_item<3> item_ct1) {
        int tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
        int size = blocks * BUCKET_BLOCK_MEMORY; 
	int sum = 0; 

	for (int i = tid; i < size; i += DIVISIONS) {
		int x = d_prefixoffsets[i]; 
		d_prefixoffsets[i] = sum; 
		sum += x; 
	}

	d_offsets[tid] = sum; 
}

void
bucketsort(float *input, int *indice, float *output, int size, unsigned int *d_prefixoffsets, 
		   unsigned int *l_offsets, sycl::nd_item<3> item_ct1,
		   volatile unsigned int *s_offset)
{

        int prefixBase = item_ct1.get_group(2) * BUCKET_BLOCK_MEMORY;
    const int warpBase =
        (item_ct1.get_local_id(2) >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads =
        item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);
        for (int i = item_ct1.get_local_id(2); i < BUCKET_BLOCK_MEMORY;
             i += item_ct1.get_local_range().get(2))
                s_offset[i] = l_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i];

        item_ct1.barrier();

        for (int tid =
                 item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
             tid < size; tid += numThreads) {

                float elem = input[tid]; 
		int id = indice[tid]; 

		output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS)] = elem;
		int test = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);

	}
}

#endif 
