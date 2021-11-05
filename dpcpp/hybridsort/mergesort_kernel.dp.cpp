#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

// declare texture reference for 1D float texture
dpct::image_wrapper<sycl::float4, 1> tex;
/*
DPCT1059:48: SYCL only supports 4-channel image format. Adjust the code.
*/
//dpct::image_wrapper<float, 1> txt;

sycl::float4 sortElem(sycl::float4 r) {
        sycl::float4 nr;

        nr.x() = (r.x() > r.y()) ? r.y() : r.x();
        nr.y() = (r.y() > r.x()) ? r.y() : r.x();
        nr.z() = (r.z() > r.w()) ? r.w() : r.z();
        nr.w() = (r.w() > r.z()) ? r.w() : r.z();

        r.x() = (nr.x() > nr.z()) ? nr.z() : nr.x();
        r.y() = (nr.y() > nr.w()) ? nr.w() : nr.y();
        r.z() = (nr.z() > nr.x()) ? nr.z() : nr.x();
        r.w() = (nr.w() > nr.y()) ? nr.w() : nr.y();

        nr.x() = r.x();
        nr.y() = (r.y() > r.z()) ? r.z() : r.y();
        nr.z() = (r.z() > r.y()) ? r.z() : r.y();
        nr.w() = r.w();
        return nr; 
}

sycl::float4 getLowest(sycl::float4 a, sycl::float4 b)
{
	//float4 na;
        a.x() = (a.x() < b.w()) ? a.x() : b.w();
        a.y() = (a.y() < b.z()) ? a.y() : b.z();
        a.z() = (a.z() < b.y()) ? a.z() : b.y();
        a.w() = (a.w() < b.x()) ? a.w() : b.x();
        return a; 
}

sycl::float4 getHighest(sycl::float4 a, sycl::float4 b)
{
        b.x() = (a.w() >= b.x()) ? a.w() : b.x();
        b.y() = (a.z() >= b.y()) ? a.z() : b.y();
        b.z() = (a.y() >= b.z()) ? a.y() : b.z();
        b.w() = (a.x() >= b.w()) ? a.x() : b.w();
        return b; 
}

dpct::constant_memory<int, 1> constStartAddr(DIVISIONS + 1);
dpct::constant_memory<int, 1> finalStartAddr(DIVISIONS + 1);
dpct::constant_memory<int, 1> nullElems(DIVISIONS);

void mergeSortFirst(sycl::float4 *result, int listsize,
                    sycl::nd_item<3> item_ct1,
                    dpct::image_accessor_ext<sycl::float4, 1> tex)
{
    // Block index
    int bx = item_ct1.get_group(2);
    // Thread index
    //int tx = threadIdx.x;
                if (bx * item_ct1.get_local_range().get(2) +
                        item_ct1.get_local_id(2) <
                    listsize / 4) {
                        sycl::float4 r =
                            tex.read((int)(bx * item_ct1.get_local_range(2) +
                                           item_ct1.get_local_id(2)));
                        result[bx * item_ct1.get_local_range().get(2) +
                               item_ct1.get_local_id(2)] = sortElem(r);
                }
}

void mergeSortPass(sycl::float4 *result, int nrElems, int threadsPerDiv,
                   sycl::nd_item<3> item_ct1, int *constStartAddr,
                   dpct::image_accessor_ext<sycl::float4, 1> tex)
{
        int tid = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
                  item_ct1.get_local_id(2);
        // The division to work on
	int division = tid / threadsPerDiv; 
	if(division >= DIVISIONS) return; 
	// The block within the division
	int int_tid = tid - division * threadsPerDiv; 
	int Astart = constStartAddr[division] + int_tid * nrElems; 

	int Bstart = Astart + nrElems/2;
        sycl::float4 *resStart = &(result[Astart]);

        if(Astart >= constStartAddr[division + 1]) 
		return; 
	if(Bstart >= constStartAddr[division + 1]){
		for(int i=0; i<(constStartAddr[division + 1] - Astart); i++)
		{
                        resStart[i] = tex.read(Astart + i);
                }
		return; 
	}

	int aidx = 0; 
	int bidx = 0; 
	int outidx = 0;
        sycl::float4 a, b;
        a = tex.read(Astart + aidx);
        b = tex.read(Bstart + bidx);

        while(true)//aidx < nrElems/2)// || (bidx < nrElems/2  && (Bstart + bidx < constEndAddr[division])))
	{
		/**
		 * For some reason, it's faster to do the texture fetches here than
		 * after the merge
		 */
                sycl::float4 nextA = tex.read(Astart + aidx + 1);
                sycl::float4 nextB = tex.read(Bstart + bidx + 1);

                sycl::float4 na = getLowest(a, b);
                sycl::float4 nb = getHighest(a, b);
                a = sortElem(na); 
		b = sortElem(nb); 
		// Now, a contains the lowest four elements, sorted
		resStart[outidx++] = a; 

		bool elemsLeftInA; 
		bool elemsLeftInB;

		elemsLeftInA = (aidx + 1 < nrElems/2); // Astart + aidx + 1 is allways less than division border 
		elemsLeftInB = (bidx + 1 < nrElems/2) && (Bstart + bidx + 1 < constStartAddr[division + 1]); 

		if(elemsLeftInA){
			if(elemsLeftInB){
                                if (nextA.x() < nextB.x()) {
                                    aidx += 1; a = nextA;
                                } else {
                                    bidx += 1; a = nextB;
                                }
                        }
			else {
				aidx += 1; a = nextA;
			}
		}
		else {
			if(elemsLeftInB){
				bidx += 1;  a = nextB;
			}
			else {
				break; 
			}
		}

	}
	resStart[outidx++] = b;
}

void
mergepack(float *orig, float *result, sycl::nd_item<3> item_ct1,
          int *constStartAddr, int *finalStartAddr, int *nullElems) 
{
        int idx = (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
                  item_ct1.get_local_id(2);
        int division = item_ct1.get_group(1);

        if((finalStartAddr[division] + idx) >= finalStartAddr[division + 1]) return; 
	result[finalStartAddr[division] + idx] = orig[constStartAddr[division]*4 + nullElems[division] + idx]; 
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
