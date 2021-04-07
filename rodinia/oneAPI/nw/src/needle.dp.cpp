#define LIMIT -999
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"

// includes, kernels
#include "needle_kernel.dp.cpp"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) try {
    std::string version;

    version = dpct::get_current_device().get_info<sycl::info::device::version>();
    
    std::cout << "CUDA driver version: " << version << std::endl;

    printf("WG size of kernel = %d \n", BLOCK_SIZE);
    runTest( argc, argv);
    return EXIT_SUCCESS;
}
catch (sycl::exception const &exc) {
    printf("Error checking driver version.\n");
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
    fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
    fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
    exit(1);
}

void runTest(int argc, char** argv)
{
dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();
    int max_rows, max_cols, penalty;
    int *input_itemsets, *output_itemsets, *referrence;
    int *matrix_cuda,  *referrence_cuda;
    int size;


    // the lengths of the two sequences should be able to divided by 16.
    // And at current stage  max_rows needs to equal max_cols
    if (argc == 3)
    {
        max_rows = atoi(argv[1]);
        max_cols = atoi(argv[1]);
        penalty = atoi(argv[2]);
    }
    else{
    usage(argc, argv);
    }

    if(atoi(argv[1])%16!=0){
    fprintf(stderr,"The dimension values must be a multiple of 16\n");
    exit(1);
    }


    max_rows = max_rows + 1;
    max_cols = max_cols + 1;
    referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
    output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );


    if (!input_itemsets)
        fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );


    for (int i = 0 ; i < max_cols; i++){
        for (int j = 0 ; j < max_rows; j++){
            input_itemsets[i*max_cols+j] = 0;
        }
    }

    printf("Start Needleman-Wunsch\n");

    for( int i=1; i< max_rows ; i++){    //please define your own sequence.
       input_itemsets[i*max_cols] = rand() % 10 + 1;
    }
    for( int j=1; j< max_cols ; j++){    //please define your own sequence.
       input_itemsets[j] = rand() % 10 + 1;
    }


    for (int i = 1 ; i < max_cols; i++){
        for (int j = 1 ; j < max_rows; j++){
        referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
        }
    }

    for( int i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
    for( int j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;


    size = max_cols * max_rows;
    referrence_cuda = sycl::malloc_device<int>(size, q_ct1);
    matrix_cuda = sycl::malloc_device<int>(size, q_ct1);

    q_ct1.memcpy(referrence_cuda, referrence, sizeof(int) * size).wait();
    q_ct1.memcpy(matrix_cuda, input_itemsets, sizeof(int) * size).wait();

    sycl::range<3> dimGrid(1, 1, 1);
    sycl::range<3> dimBlock(1, 1, BLOCK_SIZE);
    int block_width = ( max_cols - 1 )/BLOCK_SIZE;

    printf("Processing top-left matrix\n");
    //process top-left matrix
    for( int i = 1 ; i <= block_width ; i++){
        dimGrid[2] = i;
        dimGrid[1] = 1;

        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::range<2> temp_range_ct1(BLOCK_SIZE+1,BLOCK_SIZE+1);
            sycl::range<2> ref_range_ct1(BLOCK_SIZE, BLOCK_SIZE);

            sycl::accessor<int, 2, sycl::access::mode::read_write,
                           sycl::access::target::local>
                temp_acc_ct1(temp_range_ct1, cgh);
            sycl::accessor<int, 2, sycl::access::mode::read_write,
                           sycl::access::target::local>
                ref_acc_ct1(ref_range_ct1, cgh);

            cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                             [=](sycl::nd_item<3> item_ct1) {
                                 needle_cuda_shared_1(
                                     referrence_cuda, matrix_cuda, max_cols,
                                     penalty, i, block_width, item_ct1,
                                     dpct::accessor<int, dpct::local, 2>(
                                         temp_acc_ct1, temp_range_ct1),
                                     dpct::accessor<int, dpct::local, 2>(
                                         ref_acc_ct1, ref_range_ct1));
                             });
        });
    }
    printf("Processing bottom-right matrix\n");
    //process bottom-right matrix
    for( int i = block_width - 1  ; i >= 1 ; i--){
        dimGrid[2] = i;
        dimGrid[1] = 1;

        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::range<2> temp_range_ct1(BLOCK_SIZE+1, BLOCK_SIZE+1);
            sycl::range<2> ref_range_ct1(BLOCK_SIZE, BLOCK_SIZE);

            sycl::accessor<int, 2, sycl::access::mode::read_write,
                           sycl::access::target::local>
                temp_acc_ct1(temp_range_ct1, cgh);
            sycl::accessor<int, 2, sycl::access::mode::read_write,
                           sycl::access::target::local>
                ref_acc_ct1(ref_range_ct1, cgh);

            cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                             [=](sycl::nd_item<3> item_ct1) {
                                 needle_cuda_shared_2(
                                     referrence_cuda, matrix_cuda, max_cols,
                                     penalty, i, block_width, item_ct1,
                                     dpct::accessor<int, dpct::local, 2>(
                                         temp_acc_ct1, temp_range_ct1),
                                     dpct::accessor<int, dpct::local, 2>(
                                         ref_acc_ct1, ref_range_ct1));
                             });
        });
    }

    q_ct1.memcpy(output_itemsets, matrix_cuda, sizeof(int) * size).wait();

#define TRACEBACK
#ifdef TRACEBACK

    FILE *fpo = fopen("result.txt","w");
    fprintf(fpo, "print traceback value GPU:\n");

    for (int i = max_rows - 2,  j = max_rows - 2; i>=0 && j>=0;){
        int nw=0, n=0, w=0, traceback=0;
        if ( i == max_rows - 2 && j == max_rows - 2 )
            fprintf(fpo, "%d ", output_itemsets[ i * max_cols + j]); //print the first element
        if ( i == 0 && j == 0 )
            break;
        if ( i > 0 && j > 0 ){
            nw = output_itemsets[(i - 1) * max_cols + j - 1];
            w  = output_itemsets[ i * max_cols + j - 1 ];
            n  = output_itemsets[(i - 1) * max_cols + j];
        }
        else if ( i == 0 ){
            nw = n = LIMIT;
            w  = output_itemsets[ i * max_cols + j - 1 ];
        }
        else if ( j == 0 ){
            nw = w = LIMIT;
            n  = output_itemsets[(i - 1) * max_cols + j];
        }
        else{
        }

        //traceback = maximum(nw, w, n);
        int new_nw, new_w, new_n;
        new_nw = nw + referrence[i * max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);
        if(traceback == new_nw)
            traceback = nw;
        if(traceback == new_w)
            traceback = w;
        if(traceback == new_n)
            traceback = n;

        fprintf(fpo, "%d ", traceback);

        if( traceback == nw ) {
            i-- ;
            j--;
            continue;
        }
        else if( traceback == w ) {
            j--;
            continue;
        }
        else if( traceback == n ) {
            i--;
            continue;
        }
        else
            ;
    }

    fclose(fpo);

#endif

    sycl::free(referrence_cuda, q_ct1);
    sycl::free(matrix_cuda, q_ct1);

    free(referrence);
    free(input_itemsets);
    free(output_itemsets);

}
