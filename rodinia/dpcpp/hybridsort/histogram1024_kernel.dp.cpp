#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

///////////////////////////////////////////////////////////////////////////////
// This is nvidias histogram256 SDK example modded to do a 1024 point 
// histogram
///////////////////////////////////////////////////////////////////////////////

//Total number of possible data values
#define BIN_COUNT 1024 // Changed from 256
#define HISTOGRAM_SIZE (BIN_COUNT * sizeof(unsigned int))
//Machine warp size
#ifndef __DEVICE_EMULATION__
//G80's warp size is 32 threads
#define WARP_LOG_SIZE 5
#else
//Emulation currently doesn't execute threads in coherent groups of 32 threads,
//which effectively means warp size of 1 thread for emulation modes
#define WARP_LOG_SIZE 0
#endif
//Warps in thread block
#define  WARP_N 3
//Threads per block count
#ifdef HISTO_WG_SIZE_0
#define THREAD_N HISTO_WG_SIZE_0
#else
#define     THREAD_N (WARP_N << WARP_LOG_SIZE)
#endif

//Per-block number of elements in histograms
#define BLOCK_MEMORY (WARP_N * BIN_COUNT)
/*
DPCT1064:26: Migrated __mul24 call is used in a macro definition and is not
 * valid for all macro uses. Adjust the code.
*/
#define IMUL(a, b) sycl::mul24((int)a, (int)b)

void addData1024(volatile unsigned int *s_WarpHist, unsigned int data, unsigned int threadTag){
    unsigned int count;
    do{
        count = s_WarpHist[data] & 0x07FFFFFFU;
        count = threadTag | (count + 1);
        s_WarpHist[data] = count;
    }while(s_WarpHist[data] != count);
}


void histogram1024Kernel(unsigned int *d_Result, float *d_Data, float minimum, float maximum, int dataN,
                         sycl::nd_item<3> item_ct1,
                         volatile unsigned int *s_Hist){

    //Current global thread index
    const int globalTid =
        IMUL(item_ct1.get_group(2), item_ct1.get_local_range().get(2)) +
        item_ct1.get_local_id(2);
    //Total number of threads in the compute grid
    const int numThreads =
        IMUL(item_ct1.get_local_range().get(2), item_ct1.get_group_range(2));
    //WARP_LOG_SIZE higher bits of counter values are tagged 
    //by lower WARP_LOG_SIZE threadID bits
	// Will correctly issue warning when compiling for debug (x<<32-0)
    const unsigned int threadTag = item_ct1.get_local_id(2)
                                   << (32 - WARP_LOG_SIZE);
        //Shared memory cache for each warp in current thread block
    //Declare as volatile to prevent incorrect compiler optimizations in addPixel()

    //Current warp shared memory frame
    const int warpBase =
        IMUL(item_ct1.get_local_id(2) >> WARP_LOG_SIZE, BIN_COUNT);

    //Clear shared memory buffer for current thread block before processing
    for (int pos = item_ct1.get_local_id(2); pos < BLOCK_MEMORY;
         pos += item_ct1.get_local_range().get(2))
       s_Hist[pos] = 0;

    item_ct1.barrier();
    //Cycle through the entire data set, update subhistograms for each warp
    //Since threads in warps always execute the same instruction,
    //we are safe with the addPixel trick
    for(int pos = globalTid; pos < dataN; pos += numThreads){
        unsigned int data4 = ((d_Data[pos] - minimum)/(maximum - minimum)) * BIN_COUNT;
		addData1024(s_Hist + warpBase, data4 & 0x3FFU, threadTag);
    }

    item_ct1.barrier();
    //Merge per-warp histograms into per-block and write to global memory
    for (int pos = item_ct1.get_local_id(2); pos < BIN_COUNT;
         pos += item_ct1.get_local_range().get(2)) {
        unsigned int sum = 0;

        for(int base = 0; base < BLOCK_MEMORY; base += BIN_COUNT)
            sum += s_Hist[base + pos] & 0x07FFFFFFU;
         /*
         DPCT1039:27: The generated code assumes that "d_Result +
          * pos" points to the global memory address space. If it points to a
          * local memory address space, replace "dpct::atomic_fetch_add" with
          * "dpct::atomic_fetch_add<unsigned int,
          * sycl::access::address_space::local_space>".
         */
         sycl::atomic<unsigned int>(
             sycl::global_ptr<unsigned int>(d_Result + pos))
             .fetch_add(sum);
    }
}


//Thread block (== subhistogram) count
#define BLOCK_N 64


////////////////////////////////////////////////////////////////////////////////
// Put all kernels together
////////////////////////////////////////////////////////////////////////////////
//histogram1024kernel() results buffer
unsigned int *d_Result1024;

//Internal memory allocation
void initHistogram1024(void){
    /*
    DPCT1003:28: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    checkCudaErrors((d_Result1024 = (unsigned int *)sycl::malloc_device(
                         HISTOGRAM_SIZE, dpct::get_default_queue()),
                     0));
}

//Internal memory deallocation
void closeHistogram1024(void){
    /*
    DPCT1003:29: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_Result1024, dpct::get_default_queue()), 0));
}

//histogram1024 CPU front-end
void histogram1024GPU(
    unsigned int *h_Result,
    float *d_Data,
	float minimum,
	float maximum,
    int dataN)
{
    /*
    DPCT1003:30: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_default_queue()
                         .memset(d_Result1024, 0, HISTOGRAM_SIZE)
                         .wait(),
                     0));
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor<volatile unsigned int, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            s_Hist_acc_ct1(sycl::range<1>(3072 /*BLOCK_MEMORY*/), cgh);

        auto d_Result1024_ct0 = d_Result1024;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCK_N) *
                                               sycl::range<3>(1, 1, THREAD_N),
                                           sycl::range<3>(1, 1, THREAD_N)),
                         [=](sycl::nd_item<3> item_ct1) {
                             histogram1024Kernel(
                                 d_Result1024_ct0, d_Data, minimum, maximum,
                                 dataN, item_ct1, s_Hist_acc_ct1.get_pointer());
                         });
    });
    /*
    DPCT1003:31: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_default_queue()
                         .memcpy(h_Result, d_Result1024, HISTOGRAM_SIZE)
                         .wait(),
                     0));
}
