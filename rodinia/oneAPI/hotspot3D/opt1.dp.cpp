#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
long long get_time() {
        struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void hotspotOpt1(float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc, sycl::nd_item<3> item_ct1) 
{
    float amb_temp = 80.0;

    int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    int j = item_ct1.get_local_range().get(1) * item_ct1.get_group(1) +
            item_ct1.get_local_id(1);

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)        ? c : c - 1;
    int E = (i == nx-1)     ? c : c + 1;
    int N = (j == 0)        ? c : c - nx;
    int S = (j == ny-1)     ? c : c + nx;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < nz-1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c+xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    return;
}

void hotspot_opt1(float *p, float *tIn, float *tOut,
        int nx, int ny, int nz,
        float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    size_t s = sizeof(float) * nx * ny * nz;  
    float  *tIn_d, *tOut_d, *p_d;
    p_d = (float *)sycl::malloc_device(s, q_ct1);
    tIn_d = (float *)sycl::malloc_device(s, q_ct1);
    tOut_d = (float *)sycl::malloc_device(s, q_ct1);
    q_ct1.memcpy(tIn_d, tIn, s).wait();
    q_ct1.memcpy(p_d, p, s).wait();

    /*
    DPCT1004:0: Could not generate replacement.
    */
    //cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    sycl::range<3> block_dim(1, 4, 64);
    sycl::range<3> grid_dim(1, ny / 4, nx / 64);

    long long start = get_time();
    for (int i = 0; i < numiter; ++i) {
        /*
        DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                             [=](sycl::nd_item<3> item_ct1) {
                                 hotspotOpt1(p_d, tIn_d, tOut_d, stepDivCap, nx,
                                             ny, nz, ce, cw, cn, cs, ct, cb, cc,
                                             item_ct1);
                             });
        });
        float *t = tIn_d;
        tIn_d = tOut_d;
        tOut_d = t;
    }
    dev_ct1.queues_wait_and_throw();
    long long stop = get_time();
    float time = (float)((stop - start)/(1000.0 * 1000.0));
    printf("Time: %.3f (s)\n",time);
    q_ct1.memcpy(tOut, tOut_d, s).wait();
    sycl::free(p_d, q_ct1);
    sycl::free(tIn_d, q_ct1);
    sycl::free(tOut_d, q_ct1);
    return;
}

