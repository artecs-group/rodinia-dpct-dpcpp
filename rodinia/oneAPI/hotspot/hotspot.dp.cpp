#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#ifdef TIME_IT
#include <sys/time.h>
#endif

#ifdef RD_WG_SIZE_0_0                                                            
        #define BLOCK_SIZE RD_WG_SIZE_0_0                                        
#elif defined(RD_WG_SIZE_0)                                                      
        #define BLOCK_SIZE RD_WG_SIZE_0                                          
#elif defined(RD_WG_SIZE)                                                        
        #define BLOCK_SIZE RD_WG_SIZE                                            
#else                                                                                    
        #define BLOCK_SIZE 16                                                            
#endif                                                                                   

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)



void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file){

	int i,j, index=0;
	FILE *fp;
	char str[STR_SIZE];

	if( (fp = fopen(file, "w" )) == 0 )
          printf( "The file was not opened\n" );


	for (i=0; i < grid_rows; i++) 
	 for (j=0; j < grid_cols; j++)
	 {

		 sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
		 fputs(str,fp);
		 index++;
	 }
		
      fclose(fp);	
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file){

  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
            printf( "The file was not opened\n" );


	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1))
			fatal("invalid file format");
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);	

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
							   int border_cols,  // border offset 
							   int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx, 
                               float Ry, 
                               float Rz, 
                               float step, 
                               float time_elapsed,
                               sycl::nd_item<3> item_ct1,
                               dpct::accessor<float, dpct::local, 2> temp_on_cuda,
                               dpct::accessor<float, dpct::local, 2> power_on_cuda,
                               dpct::accessor<float, dpct::local, 2> temp_t){

         // saving temparary temperature result

        float amb_temp = 80.0;
        float step_div_Cap;
        float Rx_1,Ry_1,Rz_1;

        int bx = item_ct1.get_group(2);
        int by = item_ct1.get_group(1);

        int tx = item_ct1.get_local_id(2);
        int ty = item_ct1.get_local_id(1);

        step_div_Cap=step/Cap;
	
	Rx_1=1/Rx;
	Ry_1=1/Ry;
	Rz_1=1/Rz;
	
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+BLOCK_SIZE-1;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;

        // load data if it is within the valid input range
	int loadYidx=yidx, loadXidx=xidx;
        int index = grid_cols*loadYidx+loadXidx;
       
	if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
            power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
	}
        item_ct1.barrier();

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

        int N = ty-1;
        int S = ty+1;
        int W = tx-1;
        int E = tx+1;
        
        N = (N < validYmin) ? validYmin : N;
        S = (S > validYmax) ? validYmax : S;
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
	       	         (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 + 
		             (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 + 
		             (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
	
            }
            item_ct1.barrier();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            item_ct1.barrier();
          }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          temp_dst[index]= temp_t[ty][tx];		
      }
}

/*
   compute N time steps
*/

#ifdef TIME_IT
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#endif

int compute_tran_temp(float *MatrixPower,float *MatrixTemp[2], int col, int row, \
		int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows) 
{
        sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
        sycl::range<3> dimGrid(1, blockRows, blockCols);

        float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	float t;
        float time_elapsed;
	time_elapsed=0.001;

        int src = 1, dst = 0;
	
	for (t = 0; t < total_iterations; t+=num_iterations) {
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
                  sycl::range<2> temp_on_cuda_range_ct1(BLOCK_SIZE,
                                                        BLOCK_SIZE);
                  sycl::range<2> power_on_cuda_range_ct1(BLOCK_SIZE,
                                                         BLOCK_SIZE);
                  sycl::range<2> temp_t_range_ct1(BLOCK_SIZE,
                                                  BLOCK_SIZE);

                  sycl::accessor<float, 2, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      temp_on_cuda_acc_ct1(temp_on_cuda_range_ct1, cgh);
                  sycl::accessor<float, 2, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      power_on_cuda_acc_ct1(power_on_cuda_range_ct1, cgh);
                  sycl::accessor<float, 2, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      temp_t_acc_ct1(temp_t_range_ct1, cgh);

                  auto MatrixTemp_src_ct2 = MatrixTemp[src];
                  auto MatrixTemp_dst_ct3 = MatrixTemp[dst];

                  cgh.parallel_for(
                      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                      [=](sycl::nd_item<3> item_ct1) {
                            calculate_temp(
                                MIN(num_iterations, total_iterations - t),
                                MatrixPower, MatrixTemp_src_ct2,
                                MatrixTemp_dst_ct3, col, row, borderCols,
                                borderRows, Cap, Rx, Ry, Rz, step, time_elapsed,
                                item_ct1,
                                dpct::accessor<float, dpct::local, 2>(
                                    temp_on_cuda_acc_ct1,
                                    temp_on_cuda_range_ct1),
                                dpct::accessor<float, dpct::local, 2>(
                                    power_on_cuda_acc_ct1,
                                    power_on_cuda_range_ct1),
                                dpct::accessor<float, dpct::local, 2>(
                                    temp_t_acc_ct1, temp_t_range_ct1));
                      });
            });
        }
        return dst;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char** argv)
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{

    #ifdef TIME_IT
    long long initTime;
    long long alocTime = 0;
    long long cpinTime = 0;
    long long kernTime = 0;
    long long cpouTime = 0;
    long long freeTime = 0;
    long long aux1Time;
    long long aux2Time;
    #endif

    #ifdef TIME_IT
    aux1Time = get_time();
    #endif
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
    #ifdef TIME_IT
    aux2Time = get_time();
    initTime = aux2Time-aux1Time;
    #endif

    int size;
    int grid_rows,grid_cols;
    float *FilesavingTemp,*FilesavingPower,*MatrixOut; 
    char *tfile, *pfile, *ofile;
    
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations
	
	if (argc != 7)
		usage(argc, argv);
	if((grid_rows = atoi(argv[1]))<=0||
	   (grid_cols = atoi(argv[1]))<=0||
       (pyramid_height = atoi(argv[2]))<=0||
       (total_iterations = atoi(argv[3]))<=0)
		usage(argc, argv);
		
	tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];
	
    size=grid_rows*grid_cols;

    /* --------------- pyramid parameters --------------- */
    # define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    MatrixOut = (float *) calloc (size, sizeof(float));

    if( !FilesavingPower || !FilesavingTemp || !MatrixOut)
        fatal("unable to allocate memory");

    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\
	pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);
	
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);

    float *MatrixTemp[2], *MatrixPower;
    #ifdef TIME_IT
    aux1Time = get_time();
    #endif
    MatrixTemp[0] = sycl::malloc_device<float>(size, q_ct1);
    MatrixTemp[1] = sycl::malloc_device<float>(size, q_ct1);
    #ifdef TIME_IT
    aux2Time = get_time();
    alocTime += aux2Time-aux1Time;
    aux1Time = get_time();
    #endif
    q_ct1.memcpy(MatrixTemp[0], FilesavingTemp, sizeof(float) * size).wait();
    #ifdef TIME_IT
    aux2Time = get_time();
    cpinTime += aux2Time-aux1Time;
    aux1Time = get_time();
    #endif
    MatrixPower = sycl::malloc_device<float>(size, q_ct1);
    #ifdef TIME_IT
    aux2Time = get_time();
    alocTime += aux2Time-aux1Time;
    aux1Time = get_time();
    #endif
    q_ct1.memcpy(MatrixPower, FilesavingPower, sizeof(float) * size).wait();
    #ifdef TIME_IT
    aux2Time = get_time();
    cpinTime += aux2Time-aux1Time;
    #endif
    printf("Start computing the transient temperature\n");
    #ifdef TIME_IT
    aux1Time = get_time();
    #endif
    int ret = compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
	total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);
    #ifdef TIME_IT
    aux2Time = get_time();
    kernTime += aux2Time-aux1Time;
    #endif
	printf("Ending simulation\n");
    #ifdef TIME_IT
    aux1Time = get_time();
    #endif
    q_ct1.memcpy(MatrixOut, MatrixTemp[ret], sizeof(float) * size).wait();
    #ifdef TIME_IT
    aux2Time = get_time();
    cpouTime += aux2Time-aux1Time;
    #endif

    writeoutput(MatrixOut,grid_rows, grid_cols, ofile);

    #ifdef TIME_IT
    aux1Time = get_time();
    #endif
    sycl::free(MatrixPower, q_ct1);
    sycl::free(MatrixTemp[0], q_ct1);
    sycl::free(MatrixTemp[1], q_ct1);
    #ifdef TIME_IT
    aux2Time = get_time();
    freeTime += aux2Time-aux1Time;
    #endif
    free(MatrixOut);

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
}
