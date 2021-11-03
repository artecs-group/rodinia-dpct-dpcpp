/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA. Main file.
 *
 * Copyright (C) 2009 Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 * 
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "stdafx.h"
#include "cuda_helpers.h"
#include "print_helpers.h"
#include "comparison_helpers.h"
#include "stats_logger.h"
#include "load_data.h"
#include <sys/time.h>
//#include "vlc_kernel_gm32.cu"
//#include "vlc_kernel_sm32.cu"
#include "vlc_kernel_sm64huff.dp.cpp"
//#include "vlc_kernel_dpt.cu"
//#include "vlc_kernel_dptt.cu"
//#include "scan_kernel.cu"
#include "scan.dp.cpp"
#include "pack_kernels.dp.cpp"
#include "cpuencode.h"
#include <chrono>

long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks=1);

extern "C" void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, unsigned int* outdata, unsigned int *outsize, unsigned int *codewords, unsigned int* codewordlens);

int main(int argc, char* argv[]){
    if(!InitCUDA()) { return 0;	}
    unsigned int num_block_threads = 256;
    if (argc > 1)
        for (int i=1; i<argc; i++)
            runVLCTest(argv[i], num_block_threads);
    else {	runVLCTest(NULL, num_block_threads, 1024);	}
    /*
    DPCT1003:34: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_current_device().reset(), 0));
    return 0;
}

void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks) {
    printf("CUDA! Starting VLC Tests!\n");
    unsigned int num_elements; //uint num_elements = num_blocks * num_block_threads; 
    unsigned int mem_size; //uint mem_size = num_elements * sizeof(int); 
    unsigned int symbol_type_size = sizeof(int);
    //////// LOAD DATA ///////////////
    double H; // entropy
    initParams(file_name, num_block_threads, num_blocks, num_elements, mem_size, symbol_type_size);
    printf("Parameters: num_elements: %d, num_blocks: %d, num_block_threads: %d\n----------------------------\n", num_elements, num_blocks, num_block_threads);
    ////////LOAD DATA ///////////////
    uint	*sourceData =	(uint*) malloc(mem_size);
    uint	*destData   =	(uint*) malloc(mem_size);
    uint	*crefData   =	(uint*) malloc(mem_size);

    uint	*codewords	   = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
    uint	*codewordlens  = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);

    uint	*cw32 =		(uint*) malloc(mem_size);
    uint	*cw32len =	(uint*) malloc(mem_size);
    uint	*cw32idx =	(uint*) malloc(mem_size);

    uint	*cindex2=	(uint*) malloc(num_blocks*sizeof(int));

    memset(sourceData,   0, mem_size);
    memset(destData,     0, mem_size);
    memset(crefData,     0, mem_size);
    memset(cw32,         0, mem_size);
    memset(cw32len,      0, mem_size);
    memset(cw32idx,      0, mem_size);
    memset(codewords,    0, NUM_SYMBOLS*symbol_type_size);
    memset(codewordlens, 0, NUM_SYMBOLS*symbol_type_size);
    memset(cindex2, 0, num_blocks*sizeof(int));
    //////// LOAD DATA ///////////////
    loadData(file_name, sourceData, codewords, codewordlens, num_elements, mem_size, H);

    //////// LOAD DATA ///////////////

    unsigned int	*d_sourceData, *d_destData, *d_destDataPacked;
    unsigned int	*d_codewords, *d_codewordlens;
    unsigned int	*d_cw32, *d_cw32len, *d_cw32idx, *d_cindex, *d_cindex2;

    /*
    DPCT1003:39: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((d_sourceData = (unsigned int *)sycl::malloc_device(
                        mem_size, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:40: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((d_destData = (unsigned int *)sycl::malloc_device(
                        mem_size, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:41: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((d_destDataPacked = (unsigned int *)sycl::malloc_device(
                        mem_size, dpct::get_default_queue()),
                    0));

    /*
    DPCT1003:42: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (d_codewords = (unsigned int *)sycl::malloc_device(
             NUM_SYMBOLS * symbol_type_size, dpct::get_default_queue()),
         0));
    /*
    DPCT1003:43: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (d_codewordlens = (unsigned int *)sycl::malloc_device(
             NUM_SYMBOLS * symbol_type_size, dpct::get_default_queue()),
         0));

    /*
    DPCT1003:44: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((d_cw32 = (unsigned int *)sycl::malloc_device(
                        mem_size, dpct::get_default_queue()),
                    0));
    CUDA_SAFE_CALL((d_cw32len = (unsigned int *)sycl::malloc_device(
                        mem_size, dpct::get_default_queue()),
                    0));
    CUDA_SAFE_CALL((d_cw32idx = (unsigned int *)sycl::malloc_device(
                        mem_size, dpct::get_default_queue()),
                    0));

    /*
    DPCT1003:45: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((d_cindex = sycl::malloc_device<unsigned int>(
                        num_blocks, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:46: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((d_cindex2 = sycl::malloc_device<unsigned int>(
                        num_blocks, dpct::get_default_queue()),
                    0));

    /*
    DPCT1003:47: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(d_sourceData, sourceData, mem_size)
                        .wait(),
                    0));
    /*
    DPCT1003:48: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (dpct::get_default_queue()
             .memcpy(d_codewords, codewords, NUM_SYMBOLS * symbol_type_size)
             .wait(),
         0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(d_codewordlens, codewordlens,
                                NUM_SYMBOLS * symbol_type_size)
                        .wait(),
                    0));
    CUDA_SAFE_CALL((
        dpct::get_default_queue().memcpy(d_destData, destData, mem_size).wait(),
        0));

    sycl::range<3> grid_size(1, 1, num_blocks);
    sycl::range<3> block_size(1, 1, num_block_threads);
    unsigned int sm_size; 


    unsigned int NT = 10; //number of runs for each execution time

    //////////////////* CPU ENCODER *///////////////////////////////////
    unsigned int refbytesize;
    long long timer = get_time();
    cpu_vlc_encode((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);
    float msec = (float)((get_time() - timer)/1000.0);
    printf("CPU Encoding time (CPU): %f (ms)\n", msec);
    printf("CPU Encoded to %d [B]\n", refbytesize);
    unsigned int num_ints = refbytesize/4 + ((refbytesize%4 ==0)?0:1);
    //////////////////* END CPU *///////////////////////////////////

    //////////////////* SM64HUFF KERNEL *///////////////////////////////////
    grid_size[2] = num_blocks;
    block_size[2] = num_block_threads;
    sm_size = block_size[2] * sizeof(unsigned int);
#ifdef CACHECWLUT
    sm_size =
        2 * NUM_SYMBOLS * sizeof(int) + block_size[2] * sizeof(unsigned int);
#endif
    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    /*
    DPCT1026:35: The call to cudaEventCreate was removed, because this
     * call is redundant in DPC++.
    */
    /*
    DPCT1026:36: The call to cudaEventCreate was removed, because this
     * call is redundant in DPC++.
    */

    /*
    DPCT1012:37: Detected kernel execution time measurement pattern and
     * generated an initial code for time measurements in SYCL. You can change
     * the way time is measured depending on your goals.
    */
    start_ct1 = std::chrono::steady_clock::now();
        for (int i=0; i<NT; i++) {
            /*
            DPCT1049:49: The workgroup size passed to the SYCL
             * kernel may exceed the limit. To get the device limit, query
             * info::device::max_work_group_size. Adjust the workgroup size if
             * needed.
            */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      dpct_local_acc_ct1(sycl::range<1>(sm_size), cgh);
                  sycl::accessor<unsigned int, 0,
                                 sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      kcmax_acc_ct1(cgh);

                  cgh.parallel_for(
                      sycl::nd_range<3>(grid_size * block_size, block_size),
                      [=](sycl::nd_item<3> item_ct1) {
                            vlc_encode_kernel_sm64huff(
                                d_sourceData, d_codewords, d_codewordlens,
                                d_cw32, d_cw32len, d_cw32idx, d_destData,
                                d_cindex, item_ct1,
                                dpct_local_acc_ct1.get_pointer(),
                                kcmax_acc_ct1.get_pointer());
                      });
            }); // testedOK2
        }
    dpct::get_current_device().queues_wait_and_throw();
    /*
    DPCT1012:38: Detected kernel execution time measurement pattern and
     * generated an initial code for time measurements in SYCL. You can change
     * the way time is measured depending on your goals.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    float elapsedTime;
    elapsedTime =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

    CUT_CHECK_ERROR("Kernel execution failed\n");
    printf("GPU Encoding time (SM64HUFF): %f (ms)\n", elapsedTime/NT);
    //////////////////* END KERNEL *///////////////////////////////////

#ifdef TESTING
    unsigned int num_scan_elements = grid_size[2];
    preallocBlockSums(num_scan_elements);
    dpct::get_default_queue().memset(d_destDataPacked, 0, mem_size).wait();
    printf("Num_blocks to be passed to scan is %d.\n", num_scan_elements);
    prescanArray(d_cindex2, d_cindex, num_scan_elements);

      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_scan_elements / 16) *
                                      sycl::range<3>(1, 1, 16),
                                  sycl::range<3>(1, 1, 16)),
                [=](sycl::nd_item<3> item_ct1) {
                      pack2((unsigned int *)d_destData, d_cindex, d_cindex2,
                            (unsigned int *)d_destDataPacked,
                            num_elements / num_scan_elements, item_ct1);
                });
      });
    CUT_CHECK_ERROR("Pack2 Kernel execution failed\n");
    deallocBlockSums();

    /*
    DPCT1003:50: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(destData, d_destDataPacked, mem_size)
                        .wait(),
                    0));
    compare_vectors((unsigned int*)crefData, (unsigned int*)destData, num_ints);
#endif 

    free(sourceData); free(destData);  	free(codewords);  	free(codewordlens); free(cw32);  free(cw32len); free(crefData);
    /*
    DPCT1003:51: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(d_sourceData, dpct::get_default_queue()), 0));
        CUDA_SAFE_CALL((sycl::free(d_destData, dpct::get_default_queue()), 0));
        CUDA_SAFE_CALL(
            (sycl::free(d_destDataPacked, dpct::get_default_queue()), 0));
    /*
    DPCT1003:52: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(d_codewords, dpct::get_default_queue()), 0));
        CUDA_SAFE_CALL(
            (sycl::free(d_codewordlens, dpct::get_default_queue()), 0));
    /*
    DPCT1003:53: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(d_cw32, dpct::get_default_queue()), 0));
        CUDA_SAFE_CALL((sycl::free(d_cw32len, dpct::get_default_queue()), 0));
        CUDA_SAFE_CALL((sycl::free(d_cw32idx, dpct::get_default_queue()), 0));
    /*
    DPCT1003:54: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(d_cindex, dpct::get_default_queue()), 0));
        CUDA_SAFE_CALL((sycl::free(d_cindex2, dpct::get_default_queue()), 0));
    free(cindex2);
}

