
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "needle.h"
#include <stdio.h>


#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

int
maximum( int a,
         int b,
         int c){

int k;
if( a <= b )
k = b;
else
k = a;

if( k <=c )
return(c);
else
return(k);

}

void
needle_cuda_shared_1(   int* referrence,
                        int* matrix_cuda,
                        int cols,
                        int penalty,
                        int i,
                        int block_width,
                        sycl::nd_item<3> item_ct1,
                        dpct::accessor<int, dpct::local, 2> temp,
                        dpct::accessor<int, dpct::local, 2> ref)
{
  int bx = item_ct1.get_group(2);
  int tx = item_ct1.get_local_id(2);

  int b_index_x = bx;
  int b_index_y = i - 1 - bx;

  int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

   if (tx == 0)
          temp[tx][0] = matrix_cuda[index_nw];


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  ref[ty][tx] = referrence[index + cols * ty];

  item_ct1.barrier();

  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  item_ct1.barrier();

  temp[0][tx + 1] = matrix_cuda[index_n];

  item_ct1.barrier();

  for( int m = 0 ; m < BLOCK_SIZE ; m++){

      if ( tx <= m ){

          int t_index_x =  tx + 1;
          int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1]  - penalty,
                                                temp[t_index_y-1][t_index_x]  - penalty);



      }

      item_ct1.barrier();
    }

 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){

      if ( tx <= m){

          int t_index_x =  tx + BLOCK_SIZE - m ;
          int t_index_y =  BLOCK_SIZE - tx;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1]  - penalty,
                                                temp[t_index_y-1][t_index_x]  - penalty);

      }

      item_ct1.barrier();
  }

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

}


void
needle_cuda_shared_2(   int* referrence,
                        int* matrix_cuda,
                        int cols,
                        int penalty,
                        int i,
                        int block_width,
                        sycl::nd_item<3> item_ct1,
                        dpct::accessor<int, dpct::local, 2> temp,
                        dpct::accessor<int, dpct::local, 2> ref)
{

    int bx = item_ct1.get_group(2);
    int tx = item_ct1.get_local_id(2);

    int b_index_x = bx + block_width - i  ;
    int b_index_y = block_width - bx -1;

    int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
    int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
    int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
    int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

    for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
    ref[ty][tx] = referrence[index + cols * ty];

    item_ct1.barrier();

    if (tx == 0)
        temp[tx][0] = matrix_cuda[index_nw];

    temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

    item_ct1.barrier();

    temp[0][tx + 1] = matrix_cuda[index_n];

    item_ct1.barrier();

    for( int m = 0 ; m < BLOCK_SIZE ; m++){
        if ( tx <= m ){
            int t_index_x =  tx + 1;
            int t_index_y =  m - tx + 1;
            temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                  temp[t_index_y][t_index_x-1]  - penalty,
                                                  temp[t_index_y-1][t_index_x]  - penalty);
        }
    item_ct1.barrier();
    }

    for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
        if ( tx <= m){
            int t_index_x =  tx + BLOCK_SIZE - m ;
            int t_index_y =  BLOCK_SIZE - tx;
            temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                  temp[t_index_y][t_index_x-1]  - penalty,
                                                  temp[t_index_y-1][t_index_x]  - penalty);
        }
        item_ct1.barrier();
    }

    for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
    matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];
}
