#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

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


void 
lud_diagonal(float *m, int matrix_dim, int offset, sycl::nd_item<3> item_ct1,
             dpct::accessor<float, dpct::local, 2> shadow)
{
  int i,j;

  int array_offset = offset*matrix_dim+offset;
  for(i=0; i < BLOCK_SIZE; i++){
    shadow[i][item_ct1.get_local_id(2)] =
        m[array_offset + item_ct1.get_local_id(2)];
    array_offset += matrix_dim;
  }
  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance, if there is no access to global memory.
  */
  item_ct1.barrier();
  for(i=0; i < BLOCK_SIZE-1; i++) {

    if (item_ct1.get_local_id(2) > i) {
      for(j=0; j < i; j++)
        shadow[item_ct1.get_local_id(2)][i] -=
            shadow[item_ct1.get_local_id(2)][j] * shadow[j][i];
      shadow[item_ct1.get_local_id(2)][i] /= shadow[i][i];
    }

    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) > i) {

      for(j=0; j < i+1; j++)
        shadow[i + 1][item_ct1.get_local_id(2)] -=
            shadow[i + 1][j] * shadow[j][item_ct1.get_local_id(2)];
    }
    /*
    DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  /* 
     The first row is not modified, it
     is no need to write it back to the
     global memory

   */
  array_offset = (offset+1)*matrix_dim+offset;
  for(i=1; i < BLOCK_SIZE; i++){
    m[array_offset + item_ct1.get_local_id(2)] =
        shadow[i][item_ct1.get_local_id(2)];
    array_offset += matrix_dim;
  }
}

void
lud_perimeter(float *m, int matrix_dim, int offset, sycl::nd_item<3> item_ct1,
              dpct::accessor<float, dpct::local, 2> dia,
              dpct::accessor<float, dpct::local, 2> peri_row,
              dpct::accessor<float, dpct::local, 2> peri_col)
{

  int i,j, array_offset;
  int idx;

  if (item_ct1.get_local_id(2) < BLOCK_SIZE) {
    idx = item_ct1.get_local_id(2);

    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i][idx] =
          m[array_offset + (item_ct1.get_group(2) + 1) * BLOCK_SIZE + idx];
      array_offset += matrix_dim;
    }

  } else {
    idx = item_ct1.get_local_id(2) - BLOCK_SIZE;

    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }

    array_offset =
        (offset + (item_ct1.get_group(2) + 1) * BLOCK_SIZE) * matrix_dim +
        offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i][idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }
  
  }
  /*
  DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance, if there is no access to global memory.
  */
  item_ct1.barrier();

/* this version works ok on hardware, but not gpgpusim
 **************************************************************
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }

    
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }

    __syncthreads();
    
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }
***************************************************************/
  if (item_ct1.get_local_id(2) < BLOCK_SIZE) { // peri-row
    idx = item_ct1.get_local_id(2);
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }
  } else { //peri-col
    idx = item_ct1.get_local_id(2) - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }
  }

  /*
  DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance, if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < BLOCK_SIZE) { // peri-row
    idx = item_ct1.get_local_id(2);
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset + (item_ct1.get_group(2) + 1) * BLOCK_SIZE + idx] =
          peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx = item_ct1.get_local_id(2) - BLOCK_SIZE;
    array_offset =
        (offset + (item_ct1.get_group(2) + 1) * BLOCK_SIZE) * matrix_dim +
        offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }

}

void
lud_internal(float *m, int matrix_dim, int offset, sycl::nd_item<3> item_ct1,
             dpct::accessor<float, dpct::local, 2> peri_row,
             dpct::accessor<float, dpct::local, 2> peri_col)
{

  int i;
  float sum;

  int global_row_id = offset + (item_ct1.get_group(1) + 1) * BLOCK_SIZE;
  int global_col_id = offset + (item_ct1.get_group(2) + 1) * BLOCK_SIZE;

  peri_row[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] =
      m[(offset + item_ct1.get_local_id(1)) * matrix_dim + global_col_id +
        item_ct1.get_local_id(2)];
  peri_col[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] =
      m[(global_row_id + item_ct1.get_local_id(1)) * matrix_dim + offset +
        item_ct1.get_local_id(2)];

  /*
  DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance, if there is no access to global memory.
  */
  item_ct1.barrier();

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[item_ct1.get_local_id(1)][i] *
           peri_row[i][item_ct1.get_local_id(2)];
  m[(global_row_id + item_ct1.get_local_id(1)) * matrix_dim + global_col_id +
    item_ct1.get_local_id(2)] -= sum;
}

#ifdef TIME_IT
long long get_time1() {
        struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#endif

#ifdef TIME_IT
long long
#else
void 
#endif
lud_cuda(float *m, int matrix_dim)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
  int i=0;
  sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
  float *m_debug = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));

  #ifdef TIME_IT
  long long time2;
  long long time1 = get_time1();
  #endif

  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
      q_ct1.submit([&](sycl::handler &cgh) {
         sycl::range<2> shadow_range_ct1(BLOCK_SIZE, BLOCK_SIZE);

         sycl::accessor<float, 2, sycl::access::mode::read_write,
                        sycl::access::target::local>
             shadow_acc_ct1(shadow_range_ct1, cgh);

         cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCK_SIZE),
                                            sycl::range<3>(1, 1, BLOCK_SIZE)),
                          [=](sycl::nd_item<3> item_ct1) {
                             lud_diagonal(
                                 m, matrix_dim, i, item_ct1,
                                 dpct::accessor<float, dpct::local, 2>(
                                     shadow_acc_ct1, shadow_range_ct1));
                          });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
         sycl::range<2> dia_range_ct1(BLOCK_SIZE, BLOCK_SIZE);
         sycl::range<2> peri_row_range_ct1(BLOCK_SIZE,
                                           BLOCK_SIZE);
         sycl::range<2> peri_col_range_ct1(BLOCK_SIZE,
                                           BLOCK_SIZE);

         sycl::accessor<float, 2, sycl::access::mode::read_write,
                        sycl::access::target::local>
             dia_acc_ct1(dia_range_ct1, cgh);
         sycl::accessor<float, 2, sycl::access::mode::read_write,
                        sycl::access::target::local>
             peri_row_acc_ct1(peri_row_range_ct1, cgh);
         sycl::accessor<float, 2, sycl::access::mode::read_write,
                        sycl::access::target::local>
             peri_col_acc_ct1(peri_col_range_ct1, cgh);

         cgh.parallel_for(
             sycl::nd_range<3>(
                 sycl::range<3>(1, 1, (matrix_dim - i) / BLOCK_SIZE - 1) *
                     sycl::range<3>(1, 1, BLOCK_SIZE * 2),
                 sycl::range<3>(1, 1, BLOCK_SIZE * 2)),
             [=](sycl::nd_item<3> item_ct1) {
                lud_perimeter(m, matrix_dim, i, item_ct1,
                              dpct::accessor<float, dpct::local, 2>(
                                  dia_acc_ct1, dia_range_ct1),
                              dpct::accessor<float, dpct::local, 2>(
                                  peri_row_acc_ct1, peri_row_range_ct1),
                              dpct::accessor<float, dpct::local, 2>(
                                  peri_col_acc_ct1, peri_col_range_ct1));
             });
      });
      sycl::range<3> dimGrid(1, (matrix_dim - i) / BLOCK_SIZE - 1,
                             (matrix_dim - i) / BLOCK_SIZE - 1);
      /*
      DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
         sycl::range<2> peri_row_range_ct1(BLOCK_SIZE,
                                           BLOCK_SIZE);
         sycl::range<2> peri_col_range_ct1(BLOCK_SIZE,
                                           BLOCK_SIZE);

         sycl::accessor<float, 2, sycl::access::mode::read_write,
                        sycl::access::target::local>
             peri_row_acc_ct1(peri_row_range_ct1, cgh);
         sycl::accessor<float, 2, sycl::access::mode::read_write,
                        sycl::access::target::local>
             peri_col_acc_ct1(peri_col_range_ct1, cgh);

         cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                          [=](sycl::nd_item<3> item_ct1) {
                             lud_internal(
                                 m, matrix_dim, i, item_ct1,
                                 dpct::accessor<float, dpct::local, 2>(
                                     peri_row_acc_ct1, peri_row_range_ct1),
                                 dpct::accessor<float, dpct::local, 2>(
                                     peri_col_acc_ct1, peri_col_range_ct1));
                          });
      });
  }
   q_ct1.submit([&](sycl::handler &cgh) {
      sycl::range<2> shadow_range_ct1(BLOCK_SIZE, BLOCK_SIZE);

      sycl::accessor<float, 2, sycl::access::mode::read_write,
                     sycl::access::target::local>
          shadow_acc_ct1(shadow_range_ct1, cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCK_SIZE),
                                         sycl::range<3>(1, 1, BLOCK_SIZE)),
                       [=](sycl::nd_item<3> item_ct1) {
                          lud_diagonal(m, matrix_dim, i, item_ct1,
                                       dpct::accessor<float, dpct::local, 2>(
                                           shadow_acc_ct1, shadow_range_ct1));
                       });
   });

   #ifdef TIME_IT
   time2 = get_time1();
   return time2-time1;
   #endif
}

