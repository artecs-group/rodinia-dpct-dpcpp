//====================================================================================================100
//		DEFINE / INCLUDE
//====================================================================================================100

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kernel_fin_2.dp.cpp"
#include "kernel_ecc_2.dp.cpp"
#include "kernel_cam_2.dp.cpp"
#include "kernel_2.dp.cpp"
#include "embedded_fehlberg_7_8_2.dp.cpp"
#include "solver_2.dp.cpp"

//====================================================================================================100
//		MAIN FUNCTION
//====================================================================================================100

int work_2(int xmax, int workload) {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

        //================================================================================80
	//		VARIABLES
	//================================================================================80

	//============================================================60
	//		TIME
	//============================================================60

	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;

	time0 = get_time();

	//============================================================60
	//		COUNTERS, POINTERS
	//============================================================60

	long memory;
	int i;
	int pointer;

	//============================================================60
	//		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
	//============================================================60

	fp* y;
	fp* d_y;
	long y_mem;

	fp* x;
	fp* d_x;
	long x_mem;

	fp* params;
	fp* d_params;
	int params_mem;

	//============================================================60
	//		TEMPORARY SOLVER VARIABLES
	//============================================================60

	fp* d_com;
	int com_mem;

	fp* d_err;
	int err_mem;

	fp* d_scale;
	int scale_mem;

	fp* d_yy;
	int yy_mem;

	fp* d_initvalu_temp;
	int initvalu_temp_mem;

	fp* d_finavalu_temp;
	int finavalu_temp_mem;

	//============================================================60
	//		CUDA KERNELS EXECUTION PARAMETERS
	//============================================================60

        sycl::range<3> threads(1, 1, 1);
        sycl::range<3> blocks(1, 1, 1);
        int blocks_x;

	time1 = get_time();

	//================================================================================80
	// 	ALLOCATE MEMORY
	//================================================================================80

	//============================================================60
	//		MEMORY CHECK
	//============================================================60

	memory = workload*(xmax+1)*EQUATIONS*4;
	if(memory>1000000000){
		printf("ERROR: trying to allocate more than 1.0GB of memory, decrease workload and span parameters or change memory parameter\n");
		return 0;
	}

	//============================================================60
	// 	ALLOCATE ARRAYS
	//============================================================60

	//========================================40
	//		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
	//========================================40

	y_mem = workload * (xmax+1) * EQUATIONS * sizeof(fp);
	y= (fp *) malloc(y_mem);
        d_y = (float *)sycl::malloc_device(y_mem, q_ct1);

        x_mem = workload * (xmax+1) * sizeof(fp);
	x= (fp *) malloc(x_mem);
        d_x = (float *)sycl::malloc_device(x_mem, q_ct1);

        params_mem = workload * PARAMETERS * sizeof(fp);
	params= (fp *) malloc(params_mem);
        d_params = (float *)sycl::malloc_device(params_mem, q_ct1);

        //========================================40
	//		TEMPORARY SOLVER VARIABLES
	//========================================40

	com_mem = workload * 3 * sizeof(fp);
        d_com = (float *)sycl::malloc_device(com_mem, q_ct1);

        err_mem = workload * EQUATIONS * sizeof(fp);
        d_err = (float *)sycl::malloc_device(err_mem, q_ct1);

        scale_mem = workload * EQUATIONS * sizeof(fp);
        d_scale = (float *)sycl::malloc_device(scale_mem, q_ct1);

        yy_mem = workload * EQUATIONS * sizeof(fp);
        d_yy = (float *)sycl::malloc_device(yy_mem, q_ct1);

        initvalu_temp_mem = workload * EQUATIONS * sizeof(fp);
        d_initvalu_temp = (float *)sycl::malloc_device(initvalu_temp_mem, q_ct1);

        finavalu_temp_mem = workload * 13* EQUATIONS * sizeof(fp);
        d_finavalu_temp = (float *)sycl::malloc_device(finavalu_temp_mem, q_ct1);

        time2 = get_time();

	//================================================================================80
	// 	READ FROM FILES OR SET INITIAL VALUES
	//================================================================================80

	//========================================40
	//		X
	//========================================40

	for(i=0; i<workload; i++){
		pointer = i * (xmax+1) + 0;
		x[pointer] = 0;
	}
        q_ct1.memcpy(d_x, x, x_mem).wait();

        //========================================40
	//		Y
	//========================================40

	for(i=0; i<workload; i++){
		pointer = i*((xmax+1)*EQUATIONS) + 0*(EQUATIONS);
		read("../../data/myocyte/y.txt",
					&y[pointer],
					91,
					1,
					0);
	}
        q_ct1.memcpy(d_y, y, y_mem).wait();

        //========================================40
	//		PARAMS
	//========================================40

	for(i=0; i<workload; i++){
		pointer = i*PARAMETERS;
		read("../../data/myocyte/params.txt",
					&params[pointer],
					18,
					1,
					0);
	}
        q_ct1.memcpy(d_params, params, params_mem).wait();

        time3 = get_time();

	//================================================================================80
	//		EXECUTION IF THERE ARE MANY WORKLOADS
	//================================================================================80

	if(workload == 1){
                threads[2] = 32; // define the number of threads in the block
                threads[1] = 1;
                blocks[2] = 4; // define the number of blocks in the grid
                blocks[1] = 1;
        }
	else{
                threads[2] = NUMBER_THREADS; // define the number of threads in the block
                threads[1] = 1;
                blocks_x = workload / threads[2];
                if (workload % threads[2] !=
                    0) { // compensate for division remainder above by adding
                         // one grid
                        blocks_x = blocks_x + 1;
		}
                blocks[2] = blocks_x; // define the number of blocks in the grid
                blocks[1] = 1;
        }

        /*
        DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                            solver_2(workload, xmax, d_x, d_y, d_params, d_com,
                                     d_err, d_scale, d_yy, d_initvalu_temp,
                                     d_finavalu_temp, item_ct1);
                    });
        });

        // cudaThreadSynchronize();
	// printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

	time4 = get_time();

	//================================================================================80
	//		COPY DATA BACK TO CPU
	//================================================================================80

        q_ct1.memcpy(x, d_x, x_mem).wait();
        q_ct1.memcpy(y, d_y, y_mem).wait();

        time5 = get_time();

	//================================================================================80
	//		PRINT RESULTS (ENABLE SELECTIVELY FOR TESTING ONLY)
	//================================================================================80

	// int j, k;

	// for(i=0; i<workload; i++){
		// printf("WORKLOAD %d:\n", i);
		// for(j=0; j<(xmax+1); j++){
			// printf("\tTIME %d:\n", j);
			// for(k=0; k<EQUATIONS; k++){
				// printf("\t\ty[%d][%d][%d]=%13.10f\n", i, j, k, y[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k]);
			// }
		// }
	// }

	// for(i=0; i<workload; i++){
		// printf("WORKLOAD %d:\n", i);
		// for(j=0; j<(xmax+1); j++){
			// printf("\tTIME %d:\n", j);
				// printf("\t\tx[%d][%d]=%13.10f\n", i, j, x[i * (xmax+1) + j]);
		// }
	// }

	//================================================================================80
	//		DEALLOCATION
	//================================================================================80

	//============================================================60
	//		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
	//============================================================60

	free(y);
        sycl::free(d_y, q_ct1);

        free(x);
        sycl::free(d_x, q_ct1);

        free(params);
        sycl::free(d_params, q_ct1);

        //============================================================60
	//		TEMPORARY SOLVER VARIABLES
	//============================================================60

        sycl::free(d_com, q_ct1);

        sycl::free(d_err, q_ct1);
        sycl::free(d_scale, q_ct1);
        sycl::free(d_yy, q_ct1);

        sycl::free(d_initvalu_temp, q_ct1);
        sycl::free(d_finavalu_temp, q_ct1);

        time6= get_time();

	//================================================================================80
	//		DISPLAY TIMING
	//================================================================================80

	printf("Time spent in different stages of the application:\n");
	printf("%.12f s, %.12f % : SETUP VARIABLES\n", 															(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
	printf("%.12f s, %.12f % : ALLOCATE CPU MEMORY AND GPU MEMORY\n", 				(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
	printf("%.12f s, %.12f % : READ DATA FROM FILES, COPY TO GPU MEMORY\n", 		(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);
	printf("%.12f s, %.12f % : RUN GPU KERNEL\n", 															(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);
	printf("%.12f s, %.12f % : COPY GPU DATA TO CPU MEMORY\n", 								(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);
	printf("%.12f s, %.12f % : FREE MEMORY\n", 																(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time6-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", 																											(float) (time6-time0) / 1000000);

//====================================================================================================100
//		END OF FILE
//====================================================================================================100

	return 0;

}
 
 
