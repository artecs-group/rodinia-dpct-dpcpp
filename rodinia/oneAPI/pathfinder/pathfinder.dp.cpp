#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#ifdef TIME_IT
#include <sys/time.h>
#endif

//#define BLOCK_SIZE 256
//#define STR_SIZE 256
#define BLOCK_SIZE 128
#define STR_SIZE 128
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

//#define BENCH_PRINT

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

//#define BENCH_PRINT


void
init(int argc, char** argv)
{
	if(argc==4){

		cols = atoi(argv[1]);

		rows = atoi(argv[2]);

                pyramid_height=atoi(argv[3]);
	}else{
                printf("Usage: dynproc row_len col_len pyramid_height\n");
                exit(0);
        }
	data = new int[rows*cols];

	wall = new int*[rows];

	for(int n=0; n<rows; n++)

		wall[n]=data+cols*n;

	result = new int[cols];

	

	int seed = M_SEED;

	srand(seed);



	for (int i = 0; i < rows; i++)

    {

        for (int j = 0; j < cols; j++)

        {

            wall[i][j] = rand() % 10;

        }

    }

#ifdef BENCH_PRINT

    for (int i = 0; i < rows; i++)

    {

        for (int j = 0; j < cols; j++)

        {

            printf("%d ",wall[i][j]) ;

        }

        printf("\n") ;

    }

#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border,
                sycl::nd_item<3> item_ct1,
                int *prev,
                int *result)
{

        int bx = item_ct1.get_group(2);
        int tx = item_ct1.get_local_id(2);

        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;
       
        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
        item_ct1.barrier(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];
	
            }
            item_ct1.barrier();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
            item_ct1.barrier(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];		
      }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
        sycl::range<3> dimBlock(1, 1, BLOCK_SIZE);
        sycl::range<3> dimGrid(1, 1, blockCols);

        int src = 1, dst = 0;
	for (int t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            /*
            DPCT1049:0: The workgroup size passed to the SYCL
             * kernel may exceed the limit. To get the device limit, query
             * info::device::max_work_group_size. Adjust the workgroup size if
             * needed.
            */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::accessor<int, 1, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      prev_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*/), cgh);
                  sycl::accessor<int, 1, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      result_acc_ct1(sycl::range<1>(256 /*BLOCK_SIZE*/), cgh);

                  auto gpuResult_src_ct2 = gpuResult[src];
                  auto gpuResult_dst_ct3 = gpuResult[dst];

                  cgh.parallel_for(
                      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                      [=](sycl::nd_item<3> item_ct1) {
                            dynproc_kernel(MIN(pyramid_height, rows - t - 1),
                                           gpuWall, gpuResult_src_ct2,
                                           gpuResult_dst_ct3, cols, rows, t,
                                           borderCols, item_ct1,
                                           prev_acc_ct1.get_pointer(),
                                           result_acc_ct1.get_pointer());
                      });
            });
        }
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    num_devices = dpct::dev_mgr::instance().device_count();
    if (num_devices > 1) dpct::dev_mgr::instance().select_device(DEVICE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

#ifdef TIME_IT
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#endif

void run(int argc, char** argv)
{
    #ifdef TIME_IT
    long long initTime = 0;
    long long alocTime = 0;
    long long cpinTime = 0;
    long long kernTime = 0;
    long long cpouTime = 0;
    long long freeTime = 0;
    long long aux1Time;
    long long aux2Time;
    #endif

    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    int *gpuWall, *gpuResult[2];
    int size = rows*cols;

    #ifdef TIME_IT
    aux1Time = get_time();
    #endif      
    gpuResult[0] = sycl::malloc_device<int>(cols, dpct::get_default_queue());
    gpuResult[1] = sycl::malloc_device<int>(cols, dpct::get_default_queue());
    gpuWall = sycl::malloc_device<int>((size - cols), dpct::get_default_queue());
    #ifdef TIME_IT
    aux2Time = get_time();
    alocTime += aux2Time-aux1Time;
    aux1Time = get_time();
    #endif
    dpct::get_default_queue()
        .memcpy(gpuResult[0], data, sizeof(int) * cols)
        .wait();
    
    dpct::get_default_queue()
        .memcpy(gpuWall, data + cols, sizeof(int) * (size - cols))
        .wait();
    #ifdef TIME_IT
    aux2Time = get_time();
    cpinTime += aux2Time-aux1Time;
    aux1Time = get_time();
    #endif

    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
	 pyramid_height, blockCols, borderCols);

    #ifdef TIME_IT
    aux2Time = get_time();
    kernTime += aux2Time-aux1Time;
    aux1Time = get_time();
    #endif
    dpct::get_default_queue()
        .memcpy(result, gpuResult[final_ret], sizeof(int) * cols)
        .wait();
    #ifdef TIME_IT
    aux2Time = get_time();
    cpouTime += aux2Time-aux1Time;
    #endif

#ifdef BENCH_PRINT

    for (int i = 0; i < cols; i++)

            printf("%d ",data[i]) ;

    printf("\n") ;

    for (int i = 0; i < cols; i++)

            printf("%d ",result[i]) ;

    printf("\n") ;

#endif
    #ifdef TIME_IT
    aux1Time = get_time();
    #endif
    sycl::free(gpuWall, dpct::get_default_queue());
    sycl::free(gpuResult[0], dpct::get_default_queue());
    sycl::free(gpuResult[1], dpct::get_default_queue());
    #ifdef TIME_IT
    aux2Time = get_time();
    freeTime += aux2Time-aux1Time;
    #endif

    #ifdef TIME_IT
    long long totalTime = initTime + alocTime + cpinTime + kernTime + cpouTime + freeTime;
	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

	printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",	(float) initTime / 1000000, (float) initTime / (float) totalTime * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 					(float) alocTime / 1000000, (float) alocTime / (float) totalTime * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",					(float) cpinTime / 1000000, (float) cpinTime / (float) totalTime * 100);

	printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) kernTime / 1000000, (float) kernTime / (float) totalTime * 100);

	printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",				(float) cpouTime / 1000000, (float) cpouTime / (float) totalTime * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 					(float) freeTime / 1000000, (float) freeTime / (float) totalTime * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) totalTime / 1000000);
	#endif

    delete [] data;
    delete [] wall;
    delete [] result;

}

