////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mergesort.dp.hpp"
#include "mergesort_kernel.dp.cpp"
////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE	256
#define ROW_LENGTH	BLOCKSIZE * 4
#define ROWS		4096

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////
sycl::float4 *runMergeSort(int listsize, int divisions,
                           sycl::float4 *d_origList, sycl::float4 *d_resultList,
                           int *sizes, int *nullElements,
                           unsigned int *origOffsets)
{
	int *startaddr = (int *)malloc((divisions + 1)*sizeof(int)); 
	int largestSize = -1; 
	startaddr[0] = 0; 
	for(int i=1; i<=divisions; i++)
	{
		startaddr[i] = startaddr[i-1] + sizes[i-1];
		if(sizes[i-1] > largestSize) largestSize = sizes[i-1]; 
	}
	largestSize *= 4; 

	// Setup texture
        dpct::image_channel channelDesc = dpct::image_channel(
            32, 32, 32, 32, dpct::image_channel_data_type::fp);
        tex.set(sycl::addressing_mode::repeat);
        tex.set(sycl::addressing_mode::repeat);
        tex.set(sycl::filtering_mode::nearest);
        tex.set(sycl::coordinate_normalization_mode::unnormalized);

        ////////////////////////////////////////////////////////////////////////////
	// First sort all float4 elements internally
	////////////////////////////////////////////////////////////////////////////
	#ifdef MERGE_WG_SIZE_0
	const int THREADS = MERGE_WG_SIZE_0;
	#else
	const int THREADS = 256; 
	#endif
        sycl::range<3> threads(1, 1, THREADS);
        int blocks = ((listsize/4)%THREADS == 0) ? (listsize/4)/THREADS : (listsize/4)/THREADS + 1;
        sycl::range<3> grid(1, 1, blocks);
        tex.attach(d_origList, listsize * sizeof(float), channelDesc);
        /*
	DPCT1049:49: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
	*/
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto tex_acc = tex.get_access(cgh);

        auto tex_smpl = tex.get_sampler();

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             mergeSortFirst(
                                 d_resultList, listsize, item_ct1,
                                 dpct::image_accessor_ext<sycl::float4, 1>(
                                     tex_smpl, tex_acc));
                         });
    });

        ////////////////////////////////////////////////////////////////////////////
	// Then, go level by level
	////////////////////////////////////////////////////////////////////////////
        dpct::get_default_queue()
            .memcpy(constStartAddr.get_ptr(), startaddr,
                    (divisions + 1) * sizeof(int))
            .wait();
        dpct::get_default_queue()
            .memcpy(finalStartAddr.get_ptr(), origOffsets,
                    (divisions + 1) * sizeof(int))
            .wait();
        dpct::get_default_queue()
            .memcpy(nullElems.get_ptr(), nullElements,
                    (divisions) * sizeof(int))
            .wait();
        int nrElems = 2;
	while(true){
		int floatsperthread = (nrElems*4); 
		int threadsPerDiv = (int)ceil(largestSize/(float)floatsperthread); 
		int threadsNeeded = threadsPerDiv * divisions; 
		#ifdef MERGE_WG_SIZE_1
		threads.x = MERGE_WG_SIZE_1;
		#else
                threads[2] = 208; 
		#endif
                grid[2] = ((threadsNeeded % threads[2]) == 0)
                              ? threadsNeeded / threads[2]
                              : (threadsNeeded / threads[2]) + 1;
                if (grid[2] < 8) {
                        grid[2] = 8;
                        threads[2] = ((threadsNeeded % grid[2]) == 0)
                                         ? threadsNeeded / grid[2]
                                         : (threadsNeeded / grid[2]) + 1;
                }
		// Swap orig/result list
                sycl::float4 *tempList = d_origList;
                d_origList = d_resultList; 
		d_resultList = tempList;
                tex.attach(d_origList, listsize * sizeof(float), channelDesc);
                /*
		DPCT1049:51: The workgroup size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the workgroup
                 * size if needed.
		*/
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            constStartAddr.init();

            auto constStartAddr_ptr_ct1 = constStartAddr.get_ptr();

            auto tex_acc = tex.get_access(cgh);

            auto tex_smpl = tex.get_sampler();

            cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                 mergeSortPass(
                                     d_resultList, nrElems, threadsPerDiv,
                                     item_ct1, constStartAddr_ptr_ct1,
                                     dpct::image_accessor_ext<sycl::float4, 1>(
                                         tex_smpl, tex_acc));
                             });
        });
                nrElems *= 2; 
		floatsperthread = (nrElems*4); 
		if(threadsPerDiv == 1) break; 
	}
	////////////////////////////////////////////////////////////////////////////
	// Now, get rid of the NULL elements
	////////////////////////////////////////////////////////////////////////////
	#ifdef MERGE_WG_SIZE_0
	threads.x = MERGE_WG_SIZE_0;
	#else
        threads[2] = 256; 
	#endif
        grid[2] = ((largestSize % threads[2]) == 0)
                      ? largestSize / threads[2]
                      : (largestSize / threads[2]) + 1;
        grid[1] = divisions;
        /*
	DPCT1049:50: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
	*/
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        constStartAddr.init();
        finalStartAddr.init();
        nullElems.init();

        auto constStartAddr_ptr_ct1 = constStartAddr.get_ptr();
        auto finalStartAddr_ptr_ct1 = finalStartAddr.get_ptr();
        auto nullElems_ptr_ct1 = nullElems.get_ptr();

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             mergepack(
                                 (float *)d_resultList, (float *)d_origList,
                                 item_ct1, constStartAddr_ptr_ct1,
                                 finalStartAddr_ptr_ct1, nullElems_ptr_ct1);
                         });
    });

        free(startaddr);
	return d_origList; 
}
