/***********************************************
	streamcluster_cuda.cu
	: parallelized code of streamcluster
	
	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by
	
	Shawn Sang-Ha Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science
	
***********************************************/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "streamcluster_header.dp.cpp"
#include <chrono>

using namespace std;

// AUTO-ERROR CHECK FOR ALL CUDA FUNCTIONS
/*
DPCT1009:1: SYCL uses exceptions to report errors and does not use the error
 * codes. The original code was commented out and a warning string was inserted.
 * You need to rewrite this code.
*/
#define CUDA_SAFE_CALL(call)                                                   \
                              do {                                                                         \
   int err = call;                                                            \
    } while (0)

#define MAXBLOCKS 65536
#define CUDATIME

// host memory
float *work_mem_h;
float *coord_h;

// device memory
float *work_mem_d;
float *coord_d;
int   *center_table_d;
bool  *switch_membership_d;
Point *p;

static int iter = 0;		// counter for total# of iteration


//=======================================
// Euclidean Distance
//=======================================
float
d_dist(int p1, int p2, int num, int dim, float *coord_d)
{
	float retval = 0.0;
	for(int i = 0; i < dim; i++){
		float tmp = coord_d[(i*num)+p1] - coord_d[(i*num)+p2];
		retval += tmp * tmp;
	}
	return retval;
}

//=======================================
// Kernel - Compute Cost
//=======================================
void
kernel_compute_cost(int num, int dim, long x, Point *p, int K, int stride,
					float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d,
					sycl::nd_item<3> item_ct1)
{
	// block ID and global thread ID
        const int bid = item_ct1.get_group(2) +
                        item_ct1.get_group_range(2) * item_ct1.get_group(1);
        const int tid =
            item_ct1.get_local_range().get(2) * bid + item_ct1.get_local_id(2);

        if(tid < num)
	{
		float *lower = &work_mem_d[tid*stride];
		
		// cost between this point and point[x]: euclidean distance multiplied by weight
		float x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;
		
		// if computed cost is less then original (it saves), mark it as to reassign
		if ( x_cost < p[tid].cost )
		{
			switch_membership_d[tid] = 1;
			lower[K] += x_cost - p[tid].cost;
		}
		// if computed cost is larger, save the difference
		else
		{
			lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
		}
	}
}

//=======================================
// Allocate Device Memory
//=======================================
void allocDevMem(int num, int dim) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        /*
	DPCT1003:0: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((center_table_d = sycl::malloc_device<int>(num, q_ct1), 0));
        /*
	DPCT1003:2: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL(
            (switch_membership_d = sycl::malloc_device<bool>(num, q_ct1), 0));
        /*
	DPCT1003:3: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((p = sycl::malloc_device<Point>(num, q_ct1), 0));
        /*
	DPCT1003:4: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((coord_d = sycl::malloc_device<float>(num * dim, q_ct1), 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//=======================================
// Allocate Host Memory
//=======================================
void allocHostMem(int num, int dim)
{
	coord_h	= (float*) malloc( num * dim * sizeof(float) );
}

//=======================================
// Free Device Memory
//=======================================
void freeDevMem() try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        /*
	DPCT1003:5: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((sycl::free(center_table_d, q_ct1), 0));
        /*
	DPCT1003:6: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((sycl::free(switch_membership_d, q_ct1), 0));
        /*
	DPCT1003:7: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((sycl::free(p, q_ct1), 0));
        /*
	DPCT1003:8: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((sycl::free(coord_d, q_ct1), 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//=======================================
// Free Host Memory
//=======================================
void freeHostMem()
{
	free(coord_h);
}

//=======================================
// pgain Entry - CUDA SETUP + CUDA CALL
//=======================================
float pgain(long x, Points *points, float z, long int *numcenters, int kmax,
            bool *is_center, int *center_table, bool *switch_membership,
            bool isCoordChanged, double *serial_t, double *cpu_to_gpu_t,
            double *gpu_to_cpu_t, double *alloc_t, double *kernel_t,
            double *free_t) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int THREADS_PER_BLOCK = q_ct1.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
#ifdef CUDATIME
	float tmp_t;
        sycl::event start, stop;
        std::chrono::time_point<std::chrono::steady_clock> start_ct1;
        std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
        /*
	DPCT1026:9: The call to cudaEventCreate was removed, because
         * this call is redundant in DPC++.
	*/
        /*
	DPCT1026:10: The call to cudaEventCreate was removed, because
         * this call is redundant in DPC++.
	*/

        /*
	DPCT1012:11: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        start_ct1 = std::chrono::steady_clock::now();
#endif

        int error;

        int stride	= *numcenters + 1;			// size of each work_mem segment
	int K		= *numcenters ;				// number of centers
	int num		=  points->num;				// number of points
	int dim		=  points->dim;				// number of dimension
	int nThread =  num;						// number of threads == number of data points
	
	//=========================================
	// ALLOCATE HOST MEMORY + DATA PREPARATION
	//=========================================
	work_mem_h = (float*) malloc(stride * (nThread + 1) * sizeof(float) );
	// Only on the first iteration
	if(iter == 0)
	{
		allocHostMem(num, dim);
	}
	
	// build center-index table
	int count = 0;
	for( int i=0; i<num; i++)
	{
		if( is_center[i] )
		{
			center_table[i] = count++;
		}
	}

	// Extract 'coord'
	// Only if first iteration OR coord has changed
	if(isCoordChanged || iter == 0)
	{
		for(int i=0; i<dim; i++)
		{
			for(int j=0; j<num; j++)
			{
				coord_h[ (num*i)+j ] = points->p[j].coord[i];
			}
		}
	}
	
#ifdef CUDATIME
        /*
	DPCT1012:12: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        stop_ct1 = std::chrono::steady_clock::now();
        tmp_t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                    .count();
        *serial_t += (double) tmp_t;

        /*
	DPCT1012:13: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        start_ct1 = std::chrono::steady_clock::now();
#endif

	//=======================================
	// ALLOCATE GPU MEMORY
	//=======================================
        /*
	DPCT1003:26: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((work_mem_d = sycl::malloc_device<float>(
                            stride * (nThread + 1), q_ct1),
                        0));
        // Only on the first iteration
	if( iter == 0 )
	{
		allocDevMem(num, dim);
	}
	
#ifdef CUDATIME
        /*
	DPCT1012:14: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        stop_ct1 = std::chrono::steady_clock::now();
        tmp_t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                    .count();
        *alloc_t += (double) tmp_t;

        /*
	DPCT1012:15: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        start_ct1 = std::chrono::steady_clock::now();
#endif

	//=======================================
	// CPU-TO-GPU MEMORY COPY
	//=======================================
	// Only if first iteration OR coord has changed
	if(isCoordChanged || iter == 0)
	{
                /*
		DPCT1003:27: Migrated API does not return error
                 * code. (*, 0) is inserted. You may need to rewrite this code.

                 */
                CUDA_SAFE_CALL(
                    (q_ct1.memcpy(coord_d, coord_h, num * dim * sizeof(float))
                         .wait(),
                     0));
        }
        /*
	DPCT1003:28: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL(
            (q_ct1.memcpy(center_table_d, center_table, num * sizeof(int))
                 .wait(),
             0));
        /*
	DPCT1003:29: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((q_ct1.memcpy(p, points->p, num * sizeof(Point)).wait(), 0));

        /*
	DPCT1003:30: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL(
            (q_ct1.memset((void *)switch_membership_d, 0, num * sizeof(bool))
                 .wait(),
             0));
        /*
	DPCT1003:31: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((q_ct1
                            .memset((void *)work_mem_d, 0,
                                    stride * (nThread + 1) * sizeof(float))
                            .wait(),
                        0));

#ifdef CUDATIME
        /*
	DPCT1012:16: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        stop_ct1 = std::chrono::steady_clock::now();
        tmp_t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                    .count();
        *cpu_to_gpu_t += (double) tmp_t;

        /*
	DPCT1012:17: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        start_ct1 = std::chrono::steady_clock::now();
#endif
	
	//=======================================
	// KERNEL: CALCULATE COST
	//=======================================
	// Determine the number of thread blocks in the x- and y-dimension
	int num_blocks 	 = (int) ((float) (num + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);
	int num_blocks_y = (int) ((float) (num_blocks + MAXBLOCKS - 1)  / (float) MAXBLOCKS);
	int num_blocks_x = (int) ((float) (num_blocks+num_blocks_y - 1) / (float) num_blocks_y);
        sycl::range<3> grid_size(1, num_blocks_y, num_blocks_x);

        /*
	DPCT1049:19: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
	*/
        stop = q_ct1.submit([&](sycl::handler &cgh) {
                auto p_ct3 = p;
                auto coord_d_ct6 = coord_d;
                auto work_mem_d_ct7 = work_mem_d;
                auto center_table_d_ct8 = center_table_d;
                auto switch_membership_d_ct9 = switch_membership_d;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        grid_size * sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                        sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
                    [=](sycl::nd_item<3> item_ct1) {
                            kernel_compute_cost(
                                num, dim, x, p_ct3, K, stride, coord_d_ct6,
                                work_mem_d_ct7, center_table_d_ct8,
                                switch_membership_d_ct9, item_ct1);
                    });
        });
        dev_ct1.queues_wait_and_throw();

        // error check
        /*
	DPCT1010:32: SYCL uses exceptions to report errors and does not
         * use the error codes. The call was replaced with 0. You need to
         * rewrite this code.
	*/
        error = 0;

#ifdef CUDATIME
        /*
	DPCT1012:18: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        stop.wait();
        stop_ct1 = std::chrono::steady_clock::now();
        tmp_t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                    .count();
        *kernel_t += (double) tmp_t;

        /*
	DPCT1012:20: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        start_ct1 = std::chrono::steady_clock::now();
#endif
	
	//=======================================
	// GPU-TO-CPU MEMORY COPY
	//=======================================
        /*
	DPCT1003:34: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((q_ct1
                            .memcpy(work_mem_h, work_mem_d,
                                    stride * (nThread + 1) * sizeof(float))
                            .wait(),
                        0));
        /*
	DPCT1003:35: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((q_ct1
                            .memcpy(switch_membership, switch_membership_d,
                                    num * sizeof(bool))
                            .wait(),
                        0));

#ifdef CUDATIME
        /*
	DPCT1012:21: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        stop_ct1 = std::chrono::steady_clock::now();
        tmp_t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                    .count();
        *gpu_to_cpu_t += (double) tmp_t;

        /*
	DPCT1012:22: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        start_ct1 = std::chrono::steady_clock::now();
#endif
	
	//=======================================
	// CPU (SERIAL) WORK
	//=======================================
	int number_of_centers_to_close = 0;
	float gl_cost_of_opening_x = z;
	float *gl_lower = &work_mem_h[stride * nThread];
	// compute the number of centers to close if we are to open i
	for(int i=0; i < num; i++)
	{
		if( is_center[i] )
		{
			float low = z;
		    for( int j = 0; j < num; j++ )
			{
				low += work_mem_h[ j*stride + center_table[i] ];
			}
			
		    gl_lower[center_table[i]] = low;
				
		    if ( low > 0 )
			{
				++number_of_centers_to_close;
				work_mem_h[i*stride+K] -= low;
		    }
		}
		gl_cost_of_opening_x += work_mem_h[i*stride+K];
	}

	//if opening a center at x saves cost (i.e. cost is negative) do so; otherwise, do nothing
	if ( gl_cost_of_opening_x < 0 )
	{
		for(int i = 0; i < num; i++)
		{
			bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
			if ( switch_membership[i] || close_center )
			{
				points->p[i].cost = dist(points->p[i], points->p[x], dim) * points->p[i].weight;
				points->p[i].assign = x;
			}
		}
		
		for(int i = 0; i < num; i++)
		{
			if( is_center[i] && gl_lower[center_table[i]] > 0 )
			{
				is_center[i] = false;
			}
		}
		
		if( x >= 0 && x < num)
		{
			is_center[x] = true;
		}
		*numcenters = *numcenters + 1 - number_of_centers_to_close;
	}
	else
	{
		gl_cost_of_opening_x = 0;
	}
	
	//=======================================
	// DEALLOCATE HOST MEMORY
	//=======================================
	free(work_mem_h);
	
	
#ifdef CUDATIME
        /*
	DPCT1012:23: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        stop_ct1 = std::chrono::steady_clock::now();
        tmp_t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                    .count();
        *serial_t += (double) tmp_t;

        /*
	DPCT1012:24: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        start_ct1 = std::chrono::steady_clock::now();
#endif

	//=======================================
	// DEALLOCATE GPU MEMORY
	//=======================================
        /*
	DPCT1003:36: Migrated API does not return error code. (*, 0) is
         * inserted. You may need to rewrite this code.
	*/
        CUDA_SAFE_CALL((sycl::free(work_mem_d, q_ct1), 0));

#ifdef CUDATIME
        /*
	DPCT1012:25: Detected kernel execution time measurement pattern
         * and generated an initial code for time measurements in SYCL. You can
         * change the way time is measured depending on your goals.
	*/
        stop_ct1 = std::chrono::steady_clock::now();
        tmp_t = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                    .count();
        *free_t += (double) tmp_t;
#endif
	iter++;
	return -gl_cost_of_opening_x;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
