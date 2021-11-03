#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "srad.h"
#include <stdio.h>

void
srad_cuda_1(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float q0sqr
,
		  sycl::nd_item<3> item_ct1,
		  dpct::accessor<float, dpct::local, 2> temp,
		  dpct::accessor<float, dpct::local, 2> temp_result,
		  dpct::accessor<float, dpct::local, 2> north,
		  dpct::accessor<float, dpct::local, 2> south,
		  dpct::accessor<float, dpct::local, 2> east,
		  dpct::accessor<float, dpct::local, 2> west) 
{

  //block id
  int bx = item_ct1.get_group(2);
  int by = item_ct1.get_group(1);

  //thread id
  int tx = item_ct1.get_local_id(2);
  int ty = item_ct1.get_local_id(1);

  //indices
  int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
  int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation

  //load data to shared memory
  north[ty][tx] = J_cuda[index_n]; 
  south[ty][tx] = J_cuda[index_s];
  if ( by == 0 ){
  north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx];
  } else if (by == item_ct1.get_group_range(1) - 1) {
  south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (item_ct1.get_group_range(1) - 1) +
                         BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
  }
   /*
   DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
    * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    * performance, if there is no access to global memory.
   */
   item_ct1.barrier();

  west[ty][tx] = J_cuda[index_w];
  east[ty][tx] = J_cuda[index_e];

  if ( bx == 0 ){
  west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
  } else if (bx == item_ct1.get_group_range(2) - 1) {
  east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by +
                        BLOCK_SIZE * (item_ct1.get_group_range(2) - 1) +
                        cols * ty + BLOCK_SIZE - 1];
  }

  /*
  DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
   * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
   * performance, if there is no access to global memory.
  */
  item_ct1.barrier();

  temp[ty][tx]      = J_cuda[index];

  /*
  DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
   * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
   * performance, if there is no access to global memory.
  */
  item_ct1.barrier();

   jc = temp[ty][tx];

   if ( ty == 0 && tx == 0 ){ //nw
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }	    
   else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx]  - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }

   else if ( ty == 0 ){ //n
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == BLOCK_SIZE -1 ){ //e
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == 0 ){ //w
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else{  //the data elements which are not on the borders 
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }


    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

	num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
	den  = 1 + (.25*l);
	qsqr = num/(den*den);

	// diffusion coefficent (equ 33)
	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	c = 1.0 / (1.0+den) ;

    // saturate diffusion coefficent
	if (c < 0){temp_result[ty][tx] = 0;}
	else if (c > 1) {temp_result[ty][tx] = 1;}
	else {temp_result[ty][tx] = c;}

    /*
    DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
     * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
     * performance, if there is no access to global memory.
    */
    item_ct1.barrier();

    C_cuda[index] = temp_result[ty][tx];
	E_C[index] = e;
	W_C[index] = w;
	S_C[index] = s;
	N_C[index] = n;

}

void
srad_cuda_2(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,	
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float lambda,
		  float q0sqr
,
		  sycl::nd_item<3> item_ct1,
		  dpct::accessor<float, dpct::local, 2> south_c,
		  dpct::accessor<float, dpct::local, 2> east_c,
		  dpct::accessor<float, dpct::local, 2> c_cuda_temp,
		  dpct::accessor<float, dpct::local, 2> c_cuda_result,
		  dpct::accessor<float, dpct::local, 2> temp) 
{
	//block id
        int bx = item_ct1.get_group(2);
    int by = item_ct1.get_group(1);

        //thread id
    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);

        //indices
    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
	float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation

    //load data to shared memory
	temp[ty][tx]      = J_cuda[index];

    /*
    DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
     * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
     * performance, if there is no access to global memory.
    */
    item_ct1.barrier();

        south_c[ty][tx] = C_cuda[index_s];

        if (by == item_ct1.get_group_range(1) - 1) {
        south_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * (item_ct1.get_group_range(1) - 1) +
                   BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
        }
        /*
	DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
         * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
         * better performance, if there is no access to global memory.
	*/
        item_ct1.barrier();

        east_c[ty][tx] = C_cuda[index_e];

        if (bx == item_ct1.get_group_range(2) - 1) {
        east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by +
                                BLOCK_SIZE * (item_ct1.get_group_range(2) - 1) +
                                cols * ty + BLOCK_SIZE - 1];
        }

    /*
    DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
     * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
     * performance, if there is no access to global memory.
    */
    item_ct1.barrier();

    c_cuda_temp[ty][tx]      = C_cuda[index];

    /*
    DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
     * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
     * performance, if there is no access to global memory.
    */
    item_ct1.barrier();

        cc = c_cuda_temp[ty][tx];

   if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   } 
   else if ( tx == BLOCK_SIZE -1 ){ //e
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }
   else{ //the data elements which are not on the borders 
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }

   // divergence (equ 58)
   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

   // image update (equ 61)
   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

   /*
   DPCT1065:22: Consider replacing sycl::nd_item::barrier() with
    * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    * performance, if there is no access to global memory.
   */
   item_ct1.barrier();

   J_cuda[index] = c_cuda_result[ty][tx];
    
}
