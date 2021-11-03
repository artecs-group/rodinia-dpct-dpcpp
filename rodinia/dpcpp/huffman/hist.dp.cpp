/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdio.h>
#include <chrono>

#define CHECK(ans) {gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(int code, const char *file, int line, bool abort = true)
{
}

using namespace std;

#define SIZE    (100*1024*1024)


void histo_kernel( unsigned char *buffer,
        long size,
        unsigned int *histo ,
        sycl::nd_item<3> item_ct1,
        unsigned int *temp) {

    temp[item_ct1.get_local_id(2)] = 0;
    item_ct1.barrier();

    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    int offset =
        item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);
    while (i < size) {
        sycl::atomic<unsigned int, sycl::access::address_space::local_space>(
            sycl::local_ptr<unsigned int>(&temp[buffer[i]]))
            .fetch_add(1);
        i += offset;
    }

    item_ct1.barrier();
    sycl::atomic<unsigned int>(
        sycl::global_ptr<unsigned int>(&(histo[item_ct1.get_local_id(2)])))
        .fetch_add(temp[item_ct1.get_local_id(2)]);
}

int runHisto(char* file, unsigned int* freq, unsigned int memSize, unsigned int *source) {

    FILE *f = fopen(file,"rb");
    if (!f) {perror(file); exit(1);}
    fseek(f,0,SEEK_SET);
    size_t result = fread(source,1,memSize,f);
    if(result != memSize) fputs("Cannot read input file", stderr);

    fclose(f);

    unsigned char *buffer = (unsigned char*)source;

    dpct::device_info prop;
    /*
    DPCT1003:15: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    ((dpct::dev_mgr::instance().get_device(0).get_device_info(prop), 0));
    int blocks = prop.get_max_compute_units();
    /*
    DPCT1051:16: DPC++ does not support the device property that would be
     * functionally compatible with deviceOverlap. It was migrated to true. You
     * may need to rewrite the code.
    */
    if (!true)
    {
        cout << "No overlaps, so no speedup from streams" << endl;
        return 0;
    }

    // allocate memory on the GPU for the file's data
    int partSize = memSize/32;
    int totalNum = memSize/sizeof(unsigned int);
    int partialNum = partSize/sizeof(unsigned int);

    unsigned char *dev_buffer0; 
    unsigned char *dev_buffer1;
    unsigned int *dev_histo;
    dev_buffer0 = (unsigned char *)sycl::malloc_device(
        partSize, dpct::get_default_queue());
    dev_buffer1 = (unsigned char *)sycl::malloc_device(
        partSize, dpct::get_default_queue());
    dev_histo = (unsigned int *)sycl::malloc_device(256 * sizeof(int),
                                                    dpct::get_default_queue());
    dpct::get_default_queue().memset(dev_histo, 0, 256 * sizeof(int)).wait();
    sycl::queue *stream0, *stream1;
    /*
    DPCT1003:17: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CHECK((stream0 = dpct::get_current_device().create_queue(), 0));
    CHECK((stream1 = dpct::get_current_device().create_queue(), 0));
    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    /*
    DPCT1027:18: The call to cudaEventCreate was replaced with 0, because
     * this call is redundant in DPC++.
    */
    (0);
    /*
    DPCT1027:19: The call to cudaEventCreate was replaced with 0, because
     * this call is redundant in DPC++.
    */
    (0);
    /*
    DPCT1012:20: Detected kernel execution time measurement pattern and
     * generated an initial code for time measurements in SYCL. You can change
     * the way time is measured depending on your goals.
    */
    /*
    DPCT1024:21: The original code returned the error code that was
     * further consumed by the program logic. This original code was replaced
     * with 0. You may need to rewrite the program logic consuming the error
     * code.
    */
    start_ct1 = std::chrono::steady_clock::now();
    (0);

    for(int i = 0; i < totalNum; i+=partialNum*2)
    {

        /*
        DPCT1003:22: Migrated API does not return error code. (*, 0)
         * is inserted. You may need to rewrite this code.
        */
        CHECK((stream0->memcpy(dev_buffer0, buffer + i, partSize), 0));
        /*
        DPCT1003:23: Migrated API does not return error code. (*, 0)
         * is inserted. You may need to rewrite this code.
        */
        CHECK((stream1->memcpy(dev_buffer1, buffer + i + partialNum, partSize),
               0));

        // kernel launch - 2x the number of mps gave best timing
            stream0->submit([&](sycl::handler &cgh) {
                  sycl::accessor<unsigned int, 1,
                                 sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      temp_acc_ct1(sycl::range<1>(256), cgh);

                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks * 2) *
                                            sycl::range<3>(1, 1, 256),
                                        sycl::range<3>(1, 1, 256)),
                      [=](sycl::nd_item<3> item_ct1) {
                            histo_kernel(dev_buffer0, partSize, dev_histo,
                                         item_ct1, temp_acc_ct1.get_pointer());
                      });
            });
            stream1->submit([&](sycl::handler &cgh) {
                  sycl::accessor<unsigned int, 1,
                                 sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      temp_acc_ct1(sycl::range<1>(256), cgh);

                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks * 2) *
                                            sycl::range<3>(1, 1, 256),
                                        sycl::range<3>(1, 1, 256)),
                      [=](sycl::nd_item<3> item_ct1) {
                            histo_kernel(dev_buffer1, partSize, dev_histo,
                                         item_ct1, temp_acc_ct1.get_pointer());
                      });
            });
    }
    /*
    DPCT1003:24: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CHECK((stream0->wait(), 0));
    /*
    DPCT1003:25: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    CHECK((stream1->wait(), 0));
    dpct::get_default_queue().memcpy(freq, dev_histo, 256 * sizeof(int)).wait();
    /*
    DPCT1012:26: Detected kernel execution time measurement pattern and
     * generated an initial code for time measurements in SYCL. You can change
     * the way time is measured depending on your goals.
    */
    /*
    DPCT1024:27: The original code returned the error code that was
     * further consumed by the program logic. This original code was replaced
     * with 0. You may need to rewrite the program logic consuming the error
     * code.
    */
    stop_ct1 = std::chrono::steady_clock::now();
    (0);
    /*
    DPCT1003:28: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    ((stop.wait_and_throw(), 0));
    float   elapsedTime;
    /*
    DPCT1003:29: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    ((elapsedTime =
          std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
              .count(),
      0));
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    sycl::free(dev_histo, dpct::get_default_queue());
    sycl::free(dev_buffer0, dpct::get_default_queue());
    sycl::free(dev_buffer1, dpct::get_default_queue());
    return 0;
}
