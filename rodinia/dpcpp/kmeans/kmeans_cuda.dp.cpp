#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <omp.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM

#include "kmeans_cuda_kernel.dp.cpp"

//#define BLOCK_DELTA_REDUCE
//#define BLOCK_CENTER_REDUCE

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

extern "C"
int setup(int argc, char** argv);									/* function prototype */

// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */

/* _d denotes it resides on the device */
int    *membership_new;												/* newly assignment membership */
float  *feature_d;													/* inverted data array */
float  *feature_flipped_d;											/* original (not inverted) data array */
int    *membership_d;												/* membership on the device */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *clusters_d;													/* cluster centers on the device */
float  *block_clusters_d;											/* per block calculation of cluster centers */
int    *block_deltas_d;												/* per block calculation of deltas */


/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
extern "C"
void allocateMemory(int npoints, int nfeatures, int nclusters, float **features)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 printf("\nDevice Name \t\t - %s \n\n", dev_ct1.get_device_info().get_name());
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        num_blocks = npoints / num_threads;
	if (npoints % num_threads > 0)		/* defeat truncation */
		num_blocks++;

	num_blocks_perdim = sqrt((double) num_blocks);
	while (num_blocks_perdim * num_blocks_perdim < num_blocks)	// defeat truncation (should run once)
		num_blocks_perdim++;

	num_blocks = num_blocks_perdim*num_blocks_perdim;

	/* allocate memory for memory_new[] and initialize to -1 (host) */
	membership_new = (int*) malloc(npoints * sizeof(int));
	for(int i=0;i<npoints;i++) {
		membership_new[i] = -1;
	}

	/* allocate memory for block_new_centers[] (host) */
	block_new_centers = (float *) malloc(nclusters*nfeatures*sizeof(float));
	
	/* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
        feature_flipped_d = sycl::malloc_device<float>(
            npoints * nfeatures, dpct::get_default_queue());
        dpct::get_default_queue()
            .memcpy(feature_flipped_d, features[0],
                    npoints * nfeatures * sizeof(float))
            .wait();
        feature_d = sycl::malloc_device<float>(npoints * nfeatures,
                                               dpct::get_default_queue());

        /* invert the data array (kernel execution) */
        /*
	DPCT1049:3: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
	*/
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto feature_flipped_d_ct0 = feature_flipped_d;
        auto feature_d_ct1 = feature_d;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, num_threads),
                              sycl::range<3>(1, 1, num_threads)),
            [=](sycl::nd_item<3> item_ct1) {
                invert_mapping(feature_flipped_d_ct0, feature_d_ct1, npoints,
                               nfeatures, item_ct1);
            });
    });

        /* allocate memory for membership_d[] and clusters_d[][] (device) */
        membership_d = sycl::malloc_device<int>(npoints, dpct::get_default_queue());
        clusters_d = sycl::malloc_device<float>(nclusters * nfeatures,
                                                dpct::get_default_queue());

#ifdef BLOCK_DELTA_REDUCE
	// allocate array to hold the per block deltas on the gpu side
	
	cudaMalloc((void**) &block_deltas_d, num_blocks_perdim * num_blocks_perdim * sizeof(int));
	//cudaMemcpy(block_delta_d, &delta_h, sizeof(int), cudaMemcpyHostToDevice);
#endif

#ifdef BLOCK_CENTER_REDUCE
	// allocate memory and copy to card cluster  array in which to accumulate center points for the next iteration
	cudaMalloc((void**) &block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
	//cudaMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
#endif

}
/* -------------- allocateMemory() end ------------------- */

/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
extern "C"
void deallocateMemory()
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        free(membership_new);
	free(block_new_centers);
        sycl::free(feature_d, dpct::get_default_queue());
        sycl::free(feature_flipped_d, dpct::get_default_queue());
        sycl::free(membership_d, dpct::get_default_queue());

        sycl::free(clusters_d, dpct::get_default_queue());
#ifdef BLOCK_CENTER_REDUCE
    cudaFree(block_clusters_d);
#endif
#ifdef BLOCK_DELTA_REDUCE
    cudaFree(block_deltas_d);
#endif
}
/* -------------- deallocateMemory() end ------------------- */



////////////////////////////////////////////////////////////////////////////////
// Program main																  //

int
main( int argc, char** argv) 
{
	// make sure we're running on the big card
    //dpct::dev_mgr::instance().select_device(1);
        // as done in the CUDA start/help document provided
	setup(argc, argv);    
}

//																			  //
////////////////////////////////////////////////////////////////////////////////


/* ------------------- kmeansCuda() ------------------------ */
extern "C" int // delta -- had problems when return value was of float type
kmeansCuda(float **feature,      /* in: [npoints][nfeatures] */
           int nfeatures,        /* number of attributes for each point */
           int npoints,          /* number of data points */
           int nclusters,        /* number of clusters */
           int *membership,      /* which cluster the point belongs to */
           float **clusters,     /* coordinates of cluster centers */
           int *new_centers_len, /* number of elements in each cluster */
           float **new_centers   /* sum of elements in each cluster */
           ) try {
        int delta = 0;			/* if point has moved */
	int i,j;				/* counters */

        //dpct::dev_mgr::instance().select_device(1);

        /* copy membership (host to device) */
        dpct::get_default_queue()
            .memcpy(membership_d, membership_new, npoints * sizeof(int))
            .wait();

        /* copy clusters (host to device) */
        dpct::get_default_queue()
            .memcpy(clusters_d, clusters[0],
                    nclusters * nfeatures * sizeof(float))
            .wait();

        /* set up texture */
    /*
    DPCT1059:5: SYCL only supports 4-channel image format. Adjust the
     * code.
    */
    dpct::image_channel chDesc0 = dpct::image_channel::create<sycl::float4>();
    t_features.set(sycl::filtering_mode::nearest);
    t_features.set(sycl::coordinate_normalization_mode::unnormalized);
    t_features.set_channel(chDesc0);

        if ((t_features.attach(feature_d, npoints * nfeatures * sizeof(sycl::float4),
                               chDesc0),
             0) != 0)
        printf("Couldn't bind features array to texture!\n");

        /*
	DPCT1059:6: SYCL only supports 4-channel image format. Adjust
         * the code.
	*/
        dpct::image_channel chDesc1 = dpct::image_channel::create<sycl::float4>();
    t_features_flipped.set(sycl::filtering_mode::nearest);
    t_features_flipped.set(sycl::coordinate_normalization_mode::unnormalized);
    t_features_flipped.set_channel(chDesc1);

        if ((t_features_flipped.attach(feature_flipped_d,
                                       npoints * nfeatures * sizeof(sycl::float4),
                                       chDesc1),
             0) != 0)
        printf("Couldn't bind features_flipped array to texture!\n");

        /*
	DPCT1059:7: SYCL only supports 4-channel image format. Adjust
         * the code.
	*/
        dpct::image_channel chDesc2 = dpct::image_channel::create<sycl::float4>();
    t_clusters.set(sycl::filtering_mode::nearest);
    t_clusters.set(sycl::coordinate_normalization_mode::unnormalized);
    t_clusters.set_channel(chDesc2);

        if ((t_clusters.attach(clusters_d,
                               nclusters * nfeatures * sizeof(sycl::float4), chDesc2),
             0) != 0)
        printf("Couldn't bind clusters array to texture!\n");

	/* copy clusters to constant memory */
    /*
        dpct::get_default_queue().
            .memcpy("c_clusters", clusters[0],
                    nclusters * nfeatures * sizeof(float))
            .wait();
    */
        dpct::get_default_queue().submit([&](sycl::handler &h){
            h.memcpy(c_clusters.get_ptr(), clusters[0],
                    nclusters * nfeatures * sizeof(float));
        });
        dpct::get_default_queue().wait();
    /* setup execution parameters.
	   changed to 2d (source code on NVIDIA CUDA Programming Guide) */
    sycl::range<3> grid(1, num_blocks_perdim, num_blocks_perdim);
    sycl::range<3> threads(1, 1, num_threads_perdim * num_threads_perdim);

        /* execute the kernel */
    /*
    DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed
     * the limit. To get the device limit, query
     * info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        //sycl::stream out(1024, 256, cgh);

        c_clusters.init();

        auto c_clusters_ptr_ct1 = c_clusters.get_ptr();

        auto t_features_acc = t_features.get_access(cgh);

        auto t_features_smpl = t_features.get_sampler();

        auto feature_d_ct0 = feature_d;
        auto membership_d_ct4 = membership_d;
        auto clusters_d_ct5 = clusters_d;
        auto block_clusters_d_ct6 = block_clusters_d;
        auto block_deltas_d_ct7 = block_deltas_d;
std::cout << "279\n"; // print line for debug
        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             kmeansPoint(feature_d_ct0, nfeatures, npoints,
                                         nclusters, membership_d_ct4,
                                         clusters_d_ct5, block_clusters_d_ct6,
                                         block_deltas_d_ct7, item_ct1,
                                         c_clusters_ptr_ct1,
                                         dpct::image_accessor_ext<sycl::float4, 1>(
                                             t_features_smpl, t_features_acc));
                         });
    });
std::cout << "291\n"; // print line for debug
        dpct::get_current_device().queues_wait_and_throw();

        /* copy back membership (device to host) */
        dpct::get_default_queue()
            .memcpy(membership_new, membership_d, npoints * sizeof(int))
            .wait();

#ifdef BLOCK_CENTER_REDUCE
    /*** Copy back arrays of per block sums ***/
    float * block_clusters_h = (float *) malloc(
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
        
	cudaMemcpy(block_clusters_h, block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float), 
        cudaMemcpyDeviceToHost);
#endif
#ifdef BLOCK_DELTA_REDUCE
    int * block_deltas_h = (int *) malloc(
        num_blocks_perdim * num_blocks_perdim * sizeof(int));
        
	cudaMemcpy(block_deltas_h, block_deltas_d, 
        num_blocks_perdim * num_blocks_perdim * sizeof(int), 
        cudaMemcpyDeviceToHost);
#endif
    
	/* for each point, sum data points in each cluster
	   and see if membership has changed:
	     if so, increase delta and change old membership, and update new_centers;
	     otherwise, update new_centers */
	delta = 0;
	for (i = 0; i < npoints; i++)
	{		
		int cluster_id = membership_new[i];
		new_centers_len[cluster_id]++;
		if (membership_new[i] != membership[i])
		{
#ifdef CPU_DELTA_REDUCE
			delta++;
#endif
			membership[i] = membership_new[i];
		}
#ifdef CPU_CENTER_REDUCE
		for (j = 0; j < nfeatures; j++)
		{			
			new_centers[cluster_id][j] += feature[i][j];
		}
#endif
	}
	

#ifdef BLOCK_DELTA_REDUCE	
    /*** calculate global sums from per block sums for delta and the new centers ***/    
	
	//debug
	//printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim * num_blocks_perdim);
    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		//printf("block %d delta is %d \n",i,block_deltas_h[i]);
        delta += block_deltas_h[i];
    }
        
#endif
#ifdef BLOCK_CENTER_REDUCE	
	
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			block_new_centers[j*nfeatures + k] = 0.f;
		}
	}

    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		for(int j = 0; j < nclusters;j++) {
			for(int k = 0; k < nfeatures;k++) {
				block_new_centers[j*nfeatures + k] += block_clusters_h[i * nclusters*nfeatures + j * nfeatures + k];
			}
		}
    }
	

#ifdef CPU_CENTER_REDUCE
	//debug
	/*for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			if(new_centers[j][k] >	1.001 * block_new_centers[j*nfeatures + k] || new_centers[j][k] <	0.999 * block_new_centers[j*nfeatures + k]) {
				printf("\t \t for %d:%d, normal value is %e and gpu reduced value id %e \n",j,k,new_centers[j][k],block_new_centers[j*nfeatures + k]);
			}
		}
	}*/
#endif

#ifdef BLOCK_CENTER_REDUCE
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++)
			new_centers[j][k]= block_new_centers[j*nfeatures + k];		
	}
#endif

#endif

	return delta;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
/* ------------------- kmeansCuda() end ------------------------ */    

