/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */
#include <CL/sycl.hpp>
#include "dpct/dpct.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#ifdef RD_WG_SIZE_0_0
        #define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define MAXBLOCKSIZE RD_WG_SIZE
#else
        #define MAXBLOCKSIZE 512
#endif

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_XY RD_WG_SIZE
#else
        #define BLOCK_SIZE_XY 4
#endif

int Size;
float *a, *b, *finalVec;
float *m;

FILE *fp;

void InitProblemOnce(char *filename);
void InitPerRun();
void ForwardSub();
void BackSub();
void Fan1(float *m, float *a, int Size, int t, sycl::nd_item<3> item_ct1);
void Fan2(float *m, float *a, float *b,int Size, int j1, int t,
          sycl::nd_item<3> item_ct1);
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void checkCUDAError(const char *msg);

unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
        m[i*size+j]=coe[size-1-i+j];
      }
  }


}


int main(int argc, char *argv[])
{
  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
    int verbose = 1;
    int pRes = 0;
    int i, j;
    char flag;
    if (argc < 2) {
        printf("Usage: gaussian -f filename / -s size [-q]\n\n");
        printf("-q (quiet) suppresses printing the matrix and result values.\n");
        printf("-f (filename) path of input file\n");
        printf("-s (size) size of matrix. Create matrix and rhs in this program \n");
        printf("-r (result) prints the result values. \n");
        printf("The first line of the file contains the dimension of the matrix, n.");
        printf("The second line of the file is a newline.\n");
        printf("The next n lines contain n tab separated values for the matrix.");
        printf("The next line of the file is a newline.\n");
        printf("The next line of the file is a 1xn vector with tab separated values.\n");
        printf("The next line of the file is a newline. (optional)\n");
        printf("The final line of the file is the pre-computed solution. (optional)\n");
        printf("Example: matrix4.txt:\n");
        printf("4\n");
        printf("\n");
        printf("-0.6    -0.5    0.7     0.3\n");
        printf("-0.3    -0.9    0.3     0.7\n");
        printf("-0.4    -0.5    -0.3    -0.8\n");
        printf("0.0     -0.1    0.2     0.9\n");
        printf("\n");
        printf("-0.85   -0.68   0.24    -0.53\n");
        printf("\n");
        printf("0.7     0.0     -0.4    -0.5\n");
        exit(0);
    }
    
    PrintDeviceProperties();
    //char filename[100];
    //sprintf(filename,"matrices/matrix%d.txt",size);

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 's': // platform
              i++;
              Size = atoi(argv[i]);
              printf("Create matrix internally in parse, size = %d \n", Size);

              a = (float *) malloc(Size * Size * sizeof(float));
              create_matrix(a, Size);

              b = (float *) malloc(Size * sizeof(float));
              for (j =0; j< Size; j++)
                b[j]=1.0;

              m = (float *) malloc(Size * Size * sizeof(float));
              break;
            case 'f': // platform
              i++;
              printf("Read file from %s \n", argv[i]);
              InitProblemOnce(argv[i]);
              break;
            case 'q': // quiet
              verbose = 0;
              break;
            case 'r':
                pRes = 1;
                break;

          }
      }
    }

    //InitProblemOnce(filename);
    InitPerRun();
    //begin timing
    struct timeval time_start;
    gettimeofday(&time_start, NULL);
    
    // run kernels
    ForwardSub();
    
    //end timing
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    unsigned int time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
    
    if (verbose) {
        printf("Matrix m is: \n");
        PrintMat(m, Size, Size);

        printf("Matrix a is: \n");
        PrintMat(a, Size, Size);

        printf("Array b is: \n");
        PrintAry(b, Size);
    }
    BackSub();
    if (verbose || pRes) {
        printf("The final solution is: \n");
        PrintAry(finalVec,Size);
    }
    printf("\nTime total (including memory transfers)\t%f sec\n", time_total * 1e-6);
    printf("Time for SYCL kernels:\t%f sec\n",totalKernelTime * 1e-6);
    
    /*printf("%d,%d\n",size,time_total);
    fprintf(stderr,"%d,%d\n",size,time_total);*/
    
    free(m);
    free(a);
    free(b);
}
/*------------------------------------------------------
 ** PrintDeviceProperties
 **-----------------------------------------------------
 */
void PrintDeviceProperties() try {
        dpct::device_info deviceProp;
        int nDevCount = 0;

        nDevCount = dpct::dev_mgr::instance().device_count();
        printf( "Total Device found: %d", nDevCount );  
        for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx )  
        {  
                memset( &deviceProp, 0, sizeof(deviceProp));
                /*
                DPCT1003:0: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                dpct::dev_mgr::instance()
                .get_device(nDeviceIdx)
                .get_device_info(deviceProp),

                printf("\nDevice Name \t\t - %s ", deviceProp.get_name());
                printf( "\n**************************************");
                printf("\nTotal Global Memory\t\t\t - %lu KB",
                deviceProp.get_global_mem_size() / 1024);
                /*
                DPCT1019:1: local_mem_size in SYCL is not a complete
                equivalent of sharedMemPerBlock in CUDA. You may
                need to adjust the code.
                */
                printf("\nShared memory available per block \t - "
                "%lu KB",
                deviceProp.get_local_mem_size() / 1024);
                //printf( "\nNumber of registers per thread block \t - %d", deviceProp.regsPerBlock);
                printf("\nWarp size in threads \t\t\t - %d",
                deviceProp.get_max_sub_group_size());
                //printf( "\nMemory Pitch \t\t\t\t - %zu bytes", deviceProp.memPitch );
                printf("\nMaximum threads per block \t\t - %d",
                deviceProp.get_max_work_group_size());
                //printf( "\nMaximum Thread Dimension (block) \t - %d %d %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] );
                /*
                DPCT1022:2: There is no exact match between the
                maxGridSize and the max_nd_range size. Verify the
                correctness of the code.
                */
                printf("\nMaximum Thread Dimension (grid) \t - %lu "
                "%lu %lu",
                deviceProp.get_max_nd_range_size()[0],
                deviceProp.get_max_nd_range_size()[1],
                deviceProp.get_max_nd_range_size()[2]);
                //printf( "\nTotal constant memory \t\t\t - %zu bytes", deviceProp.totalConstMem );
                /*
                DPCT1005:3: The device version is different. You
                need to rewrite this code.
                */
                
                std::cout << "\nVer " << deviceProp.get_major_version() << "." << deviceProp.get_minor_version();
                
                printf("\nClock rate \t\t\t\t - %d KHz",
                deviceProp.get_max_clock_frequency());
                //printf( "\nTexture Alignment \t\t\t - %zu bytes", deviceProp.textureAlignment );
                /*
                DPCT1051:4: DPC++ does not support the device
                property that would be functionally compatible with
                deviceOverlap. It was migrated to true. You may need
                to rewrite the code.
                */
               /*
                printf("\nDevice Overlap \t\t\t\t - %s",
                true ? "Allowed" : "Not Allowed");
                */
                printf("\nNumber of Multi processors \t\t - %d\n\n",
                deviceProp.get_max_compute_units());  
        }
}
catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
        std::exit(1);
}

/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
void InitProblemOnce(char *filename)
{
        //char *filename = argv[1];

        //printf("Enter the data file name: ");
        //scanf("%s", filename);
        //printf("The file name is: %s\n", filename);

        fp = fopen(filename, "r");

        fscanf(fp, "%d", &Size);
         
        a = (float *) malloc(Size * Size * sizeof(float));
         
        InitMat(a, Size, Size);
        //printf("The input matrix a is:\n");
        //PrintMat(a, Size, Size);
        b = (float *) malloc(Size * sizeof(float));

        InitAry(b, Size);
        //printf("The input array b is:\n");
        //PrintAry(b, Size);

         m = (float *) malloc(Size * Size * sizeof(float));
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() 
{
        int i;
        for (i=0; i<Size*Size; i++)
                        *(m+i) = 0.0;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
void Fan1(float *m_cuda, float *a_cuda, int Size, int t,
          sycl::nd_item<3> item_ct1)
{   
        //if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
        //printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

        if (item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range().get(2) >=
            Size - 1 - t) return;
        *(m_cuda +
          Size * (item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2) + t + 1) +
          t) = *(a_cuda +
                 Size * (item_ct1.get_local_range().get(2) *
                             item_ct1.get_group(2) +
                         item_ct1.get_local_id(2) + t + 1) +
                 t) /
               *(a_cuda + Size * t + t);
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */ 

void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t,
          sycl::nd_item<3> item_ct1)
{
        if (item_ct1.get_local_id(2) +
                item_ct1.get_group(2) * item_ct1.get_local_range().get(2) >=
            Size - 1 - t) return;
        if (item_ct1.get_local_id(1) +
                item_ct1.get_group(1) * item_ct1.get_local_range().get(1) >=
            Size - t) return;

        int xidx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                   item_ct1.get_local_id(2);
        int yidx = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
                   item_ct1.get_local_id(1);
        //printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);

        a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
        //a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
        if(yidx == 0){
                //printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
                //printf("xidx:%d,yidx:%d\n",xidx,yidx);
                b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
        }
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub(){
        try{
                dpct::device_ext &dev_ct1 = dpct::get_current_device();
                sycl::queue &q_ct1 = dev_ct1.default_queue();
                int t;
                float *m_cuda,*a_cuda,*b_cuda;

                // allocate memory on GPU
                m_cuda = sycl::malloc_device<float>(Size * Size, q_ct1);

                a_cuda = sycl::malloc_device<float>(Size * Size, q_ct1);

                b_cuda = sycl::malloc_device<float>(Size, q_ct1);

                // copy memory to GPU
                q_ct1.memcpy(m_cuda, m, Size * Size * sizeof(float)).wait();
                q_ct1.memcpy(a_cuda, a, Size * Size * sizeof(float)).wait();
                q_ct1.memcpy(b_cuda, b, Size * sizeof(float)).wait();

                int block_size,grid_size;

                block_size = MAXBLOCKSIZE;
                grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
                //printf("1d grid size: %d\n",grid_size);

                sycl::range<3> dimBlock(1, 1, block_size);
                sycl::range<3> dimGrid(1, 1, grid_size);
                //dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

                int blockSize2d, gridSize2d;
                blockSize2d = BLOCK_SIZE_XY;
                gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1));

                sycl::range<3> dimBlockXY(1, blockSize2d, blockSize2d);
                sycl::range<3> dimGridXY(1, gridSize2d, gridSize2d);

                // begin timing kernels
                struct timeval time_start;
                gettimeofday(&time_start, NULL);
                for (t=0; t<(Size-1); t++) {
                        /*
                        DPCT1049:7: The workgroup size passed to the SYCL kernel may
                        exceed the limit. To get the device limit, query
                        info::device::max_work_group_size. Adjust the workgroup size if
                        needed.
                        */
                        q_ct1.submit([&](sycl::handler &cgh) {
                        auto Size_ct2 = Size;

                        cgh.parallel_for(
                        sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                                [=](sycl::nd_item<3> item_ct1) {
                                        Fan1(m_cuda, a_cuda, Size_ct2, t, item_ct1);
                                });
                        });
                        dev_ct1.queues_wait_and_throw();
                        /*
                        DPCT1049:8: The workgroup size passed to the SYCL kernel may
                        exceed the limit. To get the device limit, query
                        info::device::max_work_group_size. Adjust the workgroup size if
                        needed.
                        */
                        q_ct1.submit([&](sycl::handler &cgh) {
                                auto Size_ct3 = Size;
                                auto Size_t_ct4 = Size - t;

                                cgh.parallel_for(
                                sycl::nd_range<3>(dimGridXY * dimBlockXY,
                                dimBlockXY),
                                [=](sycl::nd_item<3> item_ct1) {
                                        Fan2(m_cuda, a_cuda, b_cuda, Size_ct3,
                                                Size_t_ct4, t, item_ct1);
                                });
                        });
                        dev_ct1.queues_wait_and_throw();
                        //checkCUDAError("Fan2");
                }
                // end timing kernels
                struct timeval time_end;
                gettimeofday(&time_end, NULL);
                totalKernelTime = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);

                // copy memory back to CPU
                q_ct1.memcpy(m, m_cuda, Size * Size * sizeof(float)).wait();
                q_ct1.memcpy(a, a_cuda, Size * Size * sizeof(float)).wait();
                q_ct1.memcpy(b, b_cuda, Size * sizeof(float)).wait();
                sycl::free(m_cuda, q_ct1);
                sycl::free(a_cuda, q_ct1);
                sycl::free(b_cuda, q_ct1);
        }
        catch (sycl::exception const &e) {
                std::cerr << e.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
                std::terminate();
    }
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub()
{
        // create a new vector to hold the final answer
        finalVec = (float *) malloc(Size * sizeof(float));
        // solve "bottom up"
        int i,j;
        for(i=0;i<Size;i++){
                finalVec[Size-i-1]=b[Size-i-1];
                for(j=0;j<i;j++)
                {
                        finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
                }
                finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
        }
}

void InitMat(float *ary, int nrow, int ncol)
{
        int i, j;

        for (i=0; i<nrow; i++) {
                for (j=0; j<ncol; j++) {
                        fscanf(fp, "%f",  ary+Size*i+j);
                }
        }  
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol)
{
        int i, j;

        for (i=0; i<nrow; i++) {
                for (j=0; j<ncol; j++) {
                        printf("%8.2f ", *(ary+Size*i+j));
                }
                printf("\n");
        }
        printf("\n");
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(float *ary, int ary_size)
{
        int i;

        for (i=0; i<ary_size; i++) {
                fscanf(fp, "%f",  &ary[i]);
        }
}  

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
        int i;
        for (i=0; i<ary_size; i++) {
                printf("%.2f ", ary[i]);
        }
        printf("\n\n");
}

//void checkCUDAError(const char *msg){
    /*
    DPCT1010:9: SYCL uses exceptions to report errors and does not use the error
    codes. The call was replaced with 0. You need to rewrite this code.
    */
//    int err = 0;
//}
