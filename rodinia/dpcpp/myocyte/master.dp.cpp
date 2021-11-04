#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include "../common.hpp"
//=====================================================================
//	MAIN FUNCTION
//=====================================================================

void master(fp timeinst, fp *initvalu, fp *parameter, fp *finavalu, fp *com,
			fp *d_initvalu, fp *d_finavalu, fp *d_params, fp *d_com) {
 
	select_custom_device();
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue &q_ct1 = dev_ct1.default_queue();

        //=====================================================================
	//	VARIABLES
	//=====================================================================

	// counters
	int i;

	// offset pointers
	int initvalu_offset_ecc;																// 46 points
	int initvalu_offset_Dyad;															// 15 points
	int initvalu_offset_SL;																// 15 points
	int initvalu_offset_Cyt;																// 15 poitns

	// cuda
        sycl::range<3> threads(1, 1, 1);
        sycl::range<3> blocks(1, 1, 1);

        //=====================================================================
	//	execute ECC&CAM kernel - it runs ECC and CAMs in parallel
	//=====================================================================

	int d_initvalu_mem;
	d_initvalu_mem = EQUATIONS * sizeof(fp);
	int d_finavalu_mem;
	d_finavalu_mem = EQUATIONS * sizeof(fp);
	int d_params_mem;
	d_params_mem = PARAMETERS * sizeof(fp);
	int d_com_mem;
	d_com_mem = 3 * sizeof(fp);

        q_ct1.memcpy(d_initvalu, initvalu, d_initvalu_mem).wait();
        q_ct1.memcpy(d_params, parameter, d_params_mem).wait();

        threads[2] = NUMBER_THREADS;
        threads[1] = 1;
        blocks[2] = 2;
        blocks[1] = 1;
        /*
	DPCT1049:0: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
	*/
        q_ct1.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         kernel(timeinst, d_initvalu,
                                                d_finavalu, d_params, d_com,
                                                item_ct1);
                                 });
        });

        q_ct1.memcpy(finavalu, d_finavalu, d_finavalu_mem).wait();
        q_ct1.memcpy(com, d_com, d_com_mem).wait();

        //=====================================================================
	//	FINAL KERNEL
	//=====================================================================

	initvalu_offset_ecc = 0;												// 46 points
	initvalu_offset_Dyad = 46;											// 15 points
	initvalu_offset_SL = 61;											// 15 points
	initvalu_offset_Cyt = 76;												// 15 poitns

	kernel_fin(			initvalu,
								initvalu_offset_ecc,
								initvalu_offset_Dyad,
								initvalu_offset_SL,
								initvalu_offset_Cyt,
								parameter,
								finavalu,
								com[0],
								com[1],
								com[2]);

	//=====================================================================
	//	COMPENSATION FOR NANs and INFs
	//=====================================================================

	for(i=0; i<EQUATIONS; i++){
		if (isnan(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for NAN set rate of change to 0.0001
		}
		else if (isinf(finavalu[i]) == 1){ 
			finavalu[i] = 0.0001;												// for INF set rate of change to 0.0001
		}
	}

}
