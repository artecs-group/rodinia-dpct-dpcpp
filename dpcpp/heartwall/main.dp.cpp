//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

//======================================================================================================================================================
//	LIBRARIES
//======================================================================================================================================================

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <avilib.h>
#include <avimod.h>

#ifdef TIME_IT
#include <sys/time.h>
#endif

//======================================================================================================================================================
//	STRUCTURES, GLOBAL STRUCTURE VARIABLES
//======================================================================================================================================================

#include "define.c"
#include "../common.hpp"

params_common_change common_change;
dpct::constant_memory<params_common_change, 0> d_common_change;

params_common common;
dpct::constant_memory<params_common, 0> d_common;

params_unique unique[ALL_POINTS];								// cannot determine size dynamically so choose more than usually needed
dpct::constant_memory<params_unique, 1> d_unique(ALL_POINTS);

//======================================================================================================================================================
// KERNEL CODE
//======================================================================================================================================================

#include "kernel.dp.cpp"

//	WRITE DATA FUNCTION
//===============================================================================================================================================================================================================200

void write_data(	char* filename,
			int frameNo,
			int frames_processed,
			int endoPoints,
			int* input_a,
			int* input_b,
			int epiPoints,
			int* input_2a,
			int* input_2b){

	//================================================================================80
	//	VARIABLES
	//================================================================================80

	FILE* fid;
	int i,j;
	char c;

	//================================================================================80
	//	OPEN FILE FOR READING
	//================================================================================80

	fid = fopen(filename, "w+");
	if( fid == NULL ){
		printf( "The file was not opened for writing\n" );
		return;
	}


	//================================================================================80
	//	WRITE VALUES TO THE FILE
	//================================================================================80
      fprintf(fid, "Total AVI Frames: %d\n", frameNo);	
      fprintf(fid, "Frames Processed: %d\n", frames_processed);	
      fprintf(fid, "endoPoints: %d\n", endoPoints);
      fprintf(fid, "epiPoints: %d", epiPoints);
	for(j=0; j<frames_processed;j++)
	  {
	    fprintf(fid, "\n---Frame %d---",j);
	    fprintf(fid, "\n--endo--\n",j);
	    for(i=0; i<endoPoints; i++){
	      fprintf(fid, "%d\t", input_a[j+i*frameNo]);
	    }
	    fprintf(fid, "\n");
	    for(i=0; i<endoPoints; i++){
	      // if(input_b[j*size+i] > 2000) input_b[j*size+i]=0;
	      fprintf(fid, "%d\t", input_b[j+i*frameNo]);
	    }
	    fprintf(fid, "\n--epi--\n",j);
	    for(i=0; i<epiPoints; i++){
	      //if(input_2a[j*size_2+i] > 2000) input_2a[j*size_2+i]=0;
	      fprintf(fid, "%d\t", input_2a[j+i*frameNo]);
	    }
	    fprintf(fid, "\n");
	    for(i=0; i<epiPoints; i++){
	      //if(input_2b[j*size_2+i] > 2000) input_2b[j*size_2+i]=0;
	      fprintf(fid, "%d\t", input_2b[j+i*frameNo]);
	    }
	  }
	// 	================================================================================80
	//		CLOSE FILE
		  //	================================================================================80

	fclose(fid);

}

#ifdef TIME_IT
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
#endif

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
int main(int argc, char *argv[]) {
	#ifdef TIME_IT
	long long initT=0;
	long long alocT=0;
	long long cpinT=0;
	long long kernT=0;
	long long cpouT=0;
	long long freeT=0;
	long long aux1T;
	long long aux2T;
	#endif

	#ifdef TIME_IT
    aux1T = get_time();
    #endif
    select_custom_device();
 	dpct::device_ext &dev_ct1 = dpct::get_current_device();
 	sycl::queue &q_ct1 = dev_ct1.default_queue();
	#ifdef TIME_IT
    aux2T = get_time();
	initT = aux2T-aux1T;
    #endif

  printf("WG size of kernel = %d \n", NUMBER_THREADS);
	//======================================================================================================================================================
	//	VARIABLES
	//======================================================================================================================================================

	// CUDA kernel execution parameters
        sycl::range<3> threads(1, 1, 1);
        sycl::range<3> blocks(1, 1, 1);

        // counter
	int i;
	int frames_processed;

	// frames
	char* video_file_name;
	avi_t* frames;
	fp* frame;

	//======================================================================================================================================================
	// 	FRAME
	//======================================================================================================================================================

	if(argc!=3){
		printf("ERROR: usage: heartwall <inputfile> <num of frames>\n");
		exit(1);
	}
	
	// open movie file
 	video_file_name = argv[1];
	frames = (avi_t*)AVI_open_input_file(video_file_name, 1);														// added casting
	if (frames == NULL)  {
		   AVI_print_error((char *) "Error with AVI_open_input_file");
		   return -1;
	}

	// common
	common.no_frames = AVI_video_frames(frames);
	common.frame_rows = AVI_video_height(frames);
	common.frame_cols = AVI_video_width(frames);
	common.frame_elem = common.frame_rows * common.frame_cols;
	common.frame_mem = sizeof(fp) * common.frame_elem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
    common_change.d_frame = (float *)sycl::malloc_device(common.frame_mem, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

        //======================================================================================================================================================
	// 	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================
	
	frames_processed = atoi(argv[2]);
		if(frames_processed<0 || frames_processed>common.no_frames){
			printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", frames_processed, common.no_frames);
			return 0;
	}
	

	//======================================================================================================================================================
	//	HARDCODED INPUTS FROM MATLAB
	//======================================================================================================================================================

	//====================================================================================================
	//	CONSTANTS
	//====================================================================================================

	common.sSize = 40;
	common.tSize = 25;
	common.maxMove = 10;
	common.alpha = 0.87;

	//====================================================================================================
	//	ENDO POINTS
	//====================================================================================================

	common.endoPoints = ENDO_POINTS;
	common.endo_mem = sizeof(int) * common.endoPoints;

	common.endoRow = (int *)malloc(common.endo_mem);
	common.endoRow[ 0] = 369;
	common.endoRow[ 1] = 400;
	common.endoRow[ 2] = 429;
	common.endoRow[ 3] = 452;
	common.endoRow[ 4] = 476;
	common.endoRow[ 5] = 486;
	common.endoRow[ 6] = 479;
	common.endoRow[ 7] = 458;
	common.endoRow[ 8] = 433;
	common.endoRow[ 9] = 404;
	common.endoRow[10] = 374;
	common.endoRow[11] = 346;
	common.endoRow[12] = 318;
	common.endoRow[13] = 294;
	common.endoRow[14] = 277;
	common.endoRow[15] = 269;
	common.endoRow[16] = 275;
	common.endoRow[17] = 287;
	common.endoRow[18] = 311;
	common.endoRow[19] = 339;

	#ifdef TIME_IT
	aux1T = get_time();
	#endif
    common.d_endoRow = (int *)sycl::malloc_device(common.endo_mem, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif
	#ifdef TIME_IT
	aux1T = get_time();
	#endif
    q_ct1.memcpy(common.d_endoRow, common.endoRow, common.endo_mem).wait();
	#ifdef TIME_IT
    aux2T = get_time();
	cpinT += aux2T-aux1T;
    #endif
    common.endoCol = (int *)malloc(common.endo_mem);
	common.endoCol[ 0] = 408;
	common.endoCol[ 1] = 406;
	common.endoCol[ 2] = 397;
	common.endoCol[ 3] = 383;
	common.endoCol[ 4] = 354;
	common.endoCol[ 5] = 322;
	common.endoCol[ 6] = 294;
	common.endoCol[ 7] = 270;
	common.endoCol[ 8] = 250;
	common.endoCol[ 9] = 237;
	common.endoCol[10] = 235;
	common.endoCol[11] = 241;
	common.endoCol[12] = 254;
	common.endoCol[13] = 273;
	common.endoCol[14] = 300;
	common.endoCol[15] = 328;
	common.endoCol[16] = 356;
	common.endoCol[17] = 383;
	common.endoCol[18] = 401;
	common.endoCol[19] = 411;
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
    common.d_endoCol = (int *)sycl::malloc_device(common.endo_mem, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
	aux1T = get_time();
    #endif
    q_ct1.memcpy(common.d_endoCol, common.endoCol, common.endo_mem).wait();
	#ifdef TIME_IT
    aux2T = get_time();
	cpinT += aux2T-aux1T;
    #endif

    common.tEndoRowLoc = (int *)malloc(common.endo_mem * common.no_frames);
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
        common.d_tEndoRowLoc = (int *)sycl::malloc_device(
            common.endo_mem * common.no_frames, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

        common.tEndoColLoc = (int *)malloc(common.endo_mem * common.no_frames);
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
        common.d_tEndoColLoc = (int *)sycl::malloc_device(
            common.endo_mem * common.no_frames, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif
    //====================================================================================================
	//	EPI POINTS
	//====================================================================================================

	common.epiPoints = EPI_POINTS;
	common.epi_mem = sizeof(int) * common.epiPoints;

	common.epiRow = (int *)malloc(common.epi_mem);
	common.epiRow[ 0] = 390;
	common.epiRow[ 1] = 419;
	common.epiRow[ 2] = 448;
	common.epiRow[ 3] = 474;
	common.epiRow[ 4] = 501;
	common.epiRow[ 5] = 519;
	common.epiRow[ 6] = 535;
	common.epiRow[ 7] = 542;
	common.epiRow[ 8] = 543;
	common.epiRow[ 9] = 538;
	common.epiRow[10] = 528;
	common.epiRow[11] = 511;
	common.epiRow[12] = 491;
	common.epiRow[13] = 466;
	common.epiRow[14] = 438;
	common.epiRow[15] = 406;
	common.epiRow[16] = 376;
	common.epiRow[17] = 347;
	common.epiRow[18] = 318;
	common.epiRow[19] = 291;
	common.epiRow[20] = 275;
	common.epiRow[21] = 259;
	common.epiRow[22] = 256;
	common.epiRow[23] = 252;
	common.epiRow[24] = 252;
	common.epiRow[25] = 257;
	common.epiRow[26] = 266;
	common.epiRow[27] = 283;
	common.epiRow[28] = 305;
	common.epiRow[29] = 331;
	common.epiRow[30] = 360;
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
        common.d_epiRow = (int *)sycl::malloc_device(common.epi_mem, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
	aux1T = get_time();
    #endif
        q_ct1.memcpy(common.d_epiRow, common.epiRow, common.epi_mem).wait();
	#ifdef TIME_IT
    aux2T = get_time();
	cpinT += aux2T-aux1T;
    #endif

    common.epiCol = (int *)malloc(common.epi_mem);
	common.epiCol[ 0] = 457;
	common.epiCol[ 1] = 454;
	common.epiCol[ 2] = 446;
	common.epiCol[ 3] = 431;
	common.epiCol[ 4] = 411;
	common.epiCol[ 5] = 388;
	common.epiCol[ 6] = 361;
	common.epiCol[ 7] = 331;
	common.epiCol[ 8] = 301;
	common.epiCol[ 9] = 273;
	common.epiCol[10] = 243;
	common.epiCol[11] = 218;
	common.epiCol[12] = 196;
	common.epiCol[13] = 178;
	common.epiCol[14] = 166;
	common.epiCol[15] = 157;
	common.epiCol[16] = 155;
	common.epiCol[17] = 165;
	common.epiCol[18] = 177;
	common.epiCol[19] = 197;
	common.epiCol[20] = 218;
	common.epiCol[21] = 248;
	common.epiCol[22] = 276;
	common.epiCol[23] = 304;
	common.epiCol[24] = 333;
	common.epiCol[25] = 361;
	common.epiCol[26] = 391;
	common.epiCol[27] = 415;
	common.epiCol[28] = 434;
	common.epiCol[29] = 448;
	common.epiCol[30] = 455;
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
        common.d_epiCol = (int *)sycl::malloc_device(common.epi_mem, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
	aux1T = get_time();
    #endif
        q_ct1.memcpy(common.d_epiCol, common.epiCol, common.epi_mem).wait();
	#ifdef TIME_IT
    aux2T = get_time();
	cpinT += aux2T-aux1T;
    #endif
        common.tEpiRowLoc = (int *)malloc(common.epi_mem * common.no_frames);
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
        common.d_tEpiRowLoc = (int *)sycl::malloc_device(
            common.epi_mem * common.no_frames, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

        common.tEpiColLoc = (int *)malloc(common.epi_mem * common.no_frames);
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
        common.d_tEpiColLoc = (int *)sycl::malloc_device(
            common.epi_mem * common.no_frames, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

        //====================================================================================================
	//	ALL POINTS
	//====================================================================================================

	common.allPoints = ALL_POINTS;

	//======================================================================================================================================================
	// 	TEMPLATE SIZES
	//======================================================================================================================================================

	// common
	common.in_rows = common.tSize + 1 + common.tSize;
	common.in_cols = common.in_rows;
	common.in_elem = common.in_rows * common.in_cols;
	common.in_mem = sizeof(fp) * common.in_elem;

	//======================================================================================================================================================
	// 	CREATE ARRAY OF TEMPLATES FOR ALL POINTS
	//======================================================================================================================================================

	// common
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
        common.d_endoT = (float *)sycl::malloc_device(
            common.in_mem * common.endoPoints, q_ct1);
        common.d_epiT = (float *)sycl::malloc_device(
            common.in_mem * common.epiPoints, q_ct1);
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif
    //======================================================================================================================================================
	//	SPECIFIC TO ENDO OR EPI TO BE SET HERE
	//======================================================================================================================================================

	for(i=0; i<common.endoPoints; i++){
		unique[i].point_no = i;
		unique[i].d_Row = common.d_endoRow;
		unique[i].d_Col = common.d_endoCol;
		unique[i].d_tRowLoc = common.d_tEndoRowLoc;
		unique[i].d_tColLoc = common.d_tEndoColLoc;
		unique[i].d_T = common.d_endoT;
	}
	for(i=common.endoPoints; i<common.allPoints; i++){
		unique[i].point_no = i-common.endoPoints;
		unique[i].d_Row = common.d_epiRow;
		unique[i].d_Col = common.d_epiCol;
		unique[i].d_tRowLoc = common.d_tEpiRowLoc;
		unique[i].d_tColLoc = common.d_tEpiColLoc;
		unique[i].d_T = common.d_epiT;
	}

	//======================================================================================================================================================
	// 	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY
	//======================================================================================================================================================

	// pointers
	for(i=0; i<common.allPoints; i++){
		unique[i].in_pointer = unique[i].point_no * common.in_elem;
	}

	//======================================================================================================================================================
	// 	AREA AROUND POINT		FROM	FRAME
	//======================================================================================================================================================

	// common
	common.in2_rows = 2 * common.sSize + 1;
	common.in2_cols = 2 * common.sSize + 1;
	common.in2_elem = common.in2_rows * common.in2_cols;
	common.in2_mem = sizeof(float) * common.in2_elem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
        unique[i].d_in2 = (float *)sycl::malloc_device(common.in2_mem, q_ct1);
    }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//======================================================================================================================================================
	// 	CONVOLUTION
	//======================================================================================================================================================

	// common
	common.conv_rows = common.in_rows + common.in2_rows - 1;												// number of rows in I
	common.conv_cols = common.in_cols + common.in2_cols - 1;												// number of columns in I
	common.conv_elem = common.conv_rows * common.conv_cols;												// number of elements
	common.conv_mem = sizeof(float) * common.conv_elem;
	common.ioffset = 0;
	common.joffset = 0;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_conv = (float *)sycl::malloc_device(common.conv_mem, q_ct1);
	}
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//======================================================================================================================================================
	// 	CUMULATIVE SUM
	//======================================================================================================================================================

	//====================================================================================================
	// 	PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
	//====================================================================================================

	// common
	common.in2_pad_add_rows = common.in_rows;
	common.in2_pad_add_cols = common.in_cols;

	common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
	common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
	common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
	common.in2_pad_cumv_mem = sizeof(float) * common.in2_pad_cumv_elem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_in2_pad_cumv = (float *)sycl::malloc_device(
                    common.in2_pad_cumv_mem, q_ct1);
    }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//====================================================================================================
	// 	SELECTION
	//====================================================================================================

	// common
	common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;													// (1 to n+1)
	common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
	common.in2_pad_cumv_sel_collow = 1;
	common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
	common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
	common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
	common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
	common.in2_pad_cumv_sel_mem = sizeof(float) * common.in2_pad_cumv_sel_elem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_in2_pad_cumv_sel = (float *)sycl::malloc_device(
                    common.in2_pad_cumv_sel_mem, q_ct1);
    }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//====================================================================================================
	// 	SELECTION	2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	// common
	common.in2_pad_cumv_sel2_rowlow = 1;
	common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
	common.in2_pad_cumv_sel2_collow = 1;
	common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
	common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
	common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
	common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
	common.in2_sub_cumh_mem = sizeof(float) * common.in2_sub_cumh_elem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_in2_sub_cumh = (float *)sycl::malloc_device(
                    common.in2_sub_cumh_mem, q_ct1);
    }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//====================================================================================================
	// 	SELECTION
	//====================================================================================================

	// common
	common.in2_sub_cumh_sel_rowlow = 1;
	common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
	common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
	common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
	common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
	common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
	common.in2_sub_cumh_sel_mem = sizeof(float) * common.in2_sub_cumh_sel_elem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_in2_sub_cumh_sel = (float *)sycl::malloc_device(
                    common.in2_sub_cumh_sel_mem, q_ct1);
        }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	// common
	common.in2_sub_cumh_sel2_rowlow = 1;
	common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel2_collow = 1;
	common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
	common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
	common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
	common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
	common.in2_sub2_mem = sizeof(float) * common.in2_sub2_elem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_in2_sub2 =
                    (float *)sycl::malloc_device(common.in2_sub2_mem, q_ct1);
        }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//======================================================================================================================================================
	//	CUMULATIVE SUM 2
	//======================================================================================================================================================

	//====================================================================================================
	//	MULTIPLICATION
	//====================================================================================================

	// common
	common.in2_sqr_rows = common.in2_rows;
	common.in2_sqr_cols = common.in2_cols;
	common.in2_sqr_elem = common.in2_elem;
	common.in2_sqr_mem = common.in2_mem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_in2_sqr = (float *)sycl::malloc_device(common.in2_sqr_mem, q_ct1);
        }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	// common
	common.in2_sqr_sub2_rows = common.in2_sub2_rows;
	common.in2_sqr_sub2_cols = common.in2_sub2_cols;
	common.in2_sqr_sub2_elem = common.in2_sub2_elem;
	common.in2_sqr_sub2_mem = common.in2_sub2_mem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_in2_sqr_sub2 = (float *)sycl::malloc_device(
                    common.in2_sqr_sub2_mem, q_ct1);
        }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//======================================================================================================================================================
	//	FINAL
	//======================================================================================================================================================

	// common
	common.in_sqr_rows = common.in_rows;
	common.in_sqr_cols = common.in_cols;
	common.in_sqr_elem = common.in_elem;
	common.in_sqr_mem = common.in_mem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_in_sqr = (float *)sycl::malloc_device(common.in_sqr_mem, q_ct1);
        }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif
	//======================================================================================================================================================
	//	TEMPLATE MASK CREATE
	//======================================================================================================================================================

	// common
	common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
	common.tMask_cols = common.tMask_rows;
	common.tMask_elem = common.tMask_rows * common.tMask_cols;
	common.tMask_mem = sizeof(float) * common.tMask_elem;

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_tMask = (float *)sycl::malloc_device(common.tMask_mem, q_ct1);
        }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//======================================================================================================================================================
	//	POINT MASK INITIALIZE
	//======================================================================================================================================================

	// common
	common.mask_rows = common.maxMove;
	common.mask_cols = common.mask_rows;
	common.mask_elem = common.mask_rows * common.mask_cols;
	common.mask_mem = sizeof(float) * common.mask_elem;

	//======================================================================================================================================================
	//	MASK CONVOLUTION
	//======================================================================================================================================================

	// common
	common.mask_conv_rows = common.tMask_rows;												// number of rows in I
	common.mask_conv_cols = common.tMask_cols;												// number of columns in I
	common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;												// number of elements
	common.mask_conv_mem = sizeof(float) * common.mask_conv_elem;
	common.mask_conv_ioffset = (common.mask_rows-1)/2;
	if((common.mask_rows-1) % 2 > 0.5){
		common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
	}
	common.mask_conv_joffset = (common.mask_cols-1)/2;
	if((common.mask_cols-1) % 2 > 0.5){
		common.mask_conv_joffset = common.mask_conv_joffset + 1;
	}

	// pointers
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                unique[i].d_mask_conv =
                    (float *)sycl::malloc_device(common.mask_conv_mem, q_ct1);
        }
	#ifdef TIME_IT
    aux2T = get_time();
	alocT += aux2T-aux1T;
    #endif

	//======================================================================================================================================================
	//	KERNEL
	//======================================================================================================================================================

	//====================================================================================================
	//	THREAD BLOCK
	//====================================================================================================

	// All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
        threads[2] = NUMBER_THREADS; // define the number of threads in the block
        threads[1] = 1;
        blocks[2] = common.allPoints; // define the number of blocks in the grid
        blocks[1] = 1;

    //====================================================================================================
	//	COPY ARGUMENTS
	//====================================================================================================
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
        q_ct1.memcpy(d_common.get_ptr(), &common, sizeof(params_common)).wait();
        q_ct1.memcpy(d_unique.get_ptr(), &unique,
                    sizeof(params_unique) * ALL_POINTS)
            .wait();
	#ifdef TIME_IT
    aux2T = get_time();
	cpinT += aux2T-aux1T;
    #endif

        //====================================================================================================
	//	PRINT FRAME PROGRESS START
	//====================================================================================================

	printf("frame progress: ");
	fflush(NULL);

	//====================================================================================================
	//	LAUNCH
	//====================================================================================================

	for(common_change.frame_no=0; common_change.frame_no<frames_processed; common_change.frame_no++){

		// Extract a cropped version of the first frame from the video file
		frame = get_frame(	frames,						// pointer to video file
										common_change.frame_no,				// number of frame that needs to be returned
										0,								// cropped?
										0,								// scaled?
										1);							// converted

		// copy frame to GPU memory
		#ifdef TIME_IT
    	aux1T = get_time();
    	#endif
                q_ct1.memcpy(common_change.d_frame, frame, common.frame_mem).wait();
                q_ct1.memcpy(d_common_change.get_ptr(), &common_change,
                            sizeof(params_common_change))
                    .wait();
		#ifdef TIME_IT
    	aux2T = get_time();
		cpinT += aux2T-aux1T;
		aux1T = get_time();
    	#endif
                // launch GPU kernel
                /*
                DPCT1049:0: The workgroup size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the workgroup size if
                needed.
                */
                q_ct1.submit([&](sycl::handler &cgh) {
                        d_common_change.init();
                        d_common.init();
                        d_unique.init();

                        auto d_common_change_ptr_ct1 =
                            d_common_change.get_ptr();
                        auto d_common_ptr_ct1 = d_common.get_ptr();
                        auto d_unique_ptr_ct1 = d_unique.get_ptr();

                        sycl::accessor<float, 1, sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            in_partial_sum_acc_ct1(sycl::range<1>(51), cgh);
                        sycl::accessor<float, 1, sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            in_sqr_partial_sum_acc_ct1(sycl::range<1>(51), cgh);
                        sycl::accessor<float, 0, sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            in_final_sum_acc_ct1(cgh);
                        sycl::accessor<float, 0, sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            in_sqr_final_sum_acc_ct1(cgh);
                        sycl::accessor<float, 0, sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            denomT_acc_ct1(cgh);
                        sycl::accessor<float, 1, sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            par_max_val_acc_ct1(sycl::range<1>(131), cgh);
                        sycl::accessor<int, 1, sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            par_max_coo_acc_ct1(sycl::range<1>(131), cgh);
                        sycl::accessor<float, 1, sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            d_in_mod_temp_acc_ct1(sycl::range<1>(2601), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(blocks * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel(
                                        item_ct1, *d_common_change_ptr_ct1,
                                        *d_common_ptr_ct1, d_unique_ptr_ct1,
                                        in_partial_sum_acc_ct1.get_pointer(),
                                        in_sqr_partial_sum_acc_ct1
                                            .get_pointer(),
                                        in_final_sum_acc_ct1.get_pointer(),
                                        in_sqr_final_sum_acc_ct1.get_pointer(),
                                        denomT_acc_ct1.get_pointer(),
                                        par_max_val_acc_ct1.get_pointer(),
                                        par_max_coo_acc_ct1.get_pointer(),
                                        d_in_mod_temp_acc_ct1.get_pointer());
                            });
                });
			#ifdef TIME_IT
    		aux2T = get_time();
			kernT += aux2T-aux1T;
    		#endif
                // free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		free(frame);

		// print frame progress
		printf("%d ", common_change.frame_no);
		fflush(NULL);

	}

	//====================================================================================================
	//	PRINT FRAME PROGRESS END
	//====================================================================================================

	printf("\n");
	fflush(NULL);

	//====================================================================================================
	//	OUTPUT
	//====================================================================================================
		#ifdef TIME_IT
    	aux1T = get_time();
    	#endif
        q_ct1
            .memcpy(common.tEndoRowLoc, common.d_tEndoRowLoc,
                    common.endo_mem * common.no_frames)
            .wait();
        q_ct1
            .memcpy(common.tEndoColLoc, common.d_tEndoColLoc,
                    common.endo_mem * common.no_frames)
            .wait();

        q_ct1
            .memcpy(common.tEpiRowLoc, common.d_tEpiRowLoc,
                    common.epi_mem * common.no_frames)
            .wait();
        q_ct1
            .memcpy(common.tEpiColLoc, common.d_tEpiColLoc,
                    common.epi_mem * common.no_frames)
            .wait();
		#ifdef TIME_IT
   	 	aux2T = get_time();
		cpouT += aux2T-aux1T;
    	#endif
#ifdef OUTPUT

	//==================================================50
	//	DUMP DATA TO FILE
	//==================================================50
	write_data(	"result.txt",
			common.no_frames,
			frames_processed,		
				common.endoPoints,
				common.tEndoRowLoc,
				common.tEndoColLoc,
				common.epiPoints,
				common.tEpiRowLoc,
				common.tEpiColLoc);

	//==================================================50
	//	End
	//==================================================50

#endif



	//======================================================================================================================================================
	//	DEALLOCATION
	//======================================================================================================================================================

	//====================================================================================================
	//	COMMON
	//====================================================================================================

	// frame
		#ifdef TIME_IT
    	aux1T = get_time();
    	#endif
        sycl::free(common_change.d_frame, q_ct1);
		#ifdef TIME_IT
    	aux2T = get_time();
		freeT += aux2T-aux1T;
    	#endif

        // endo points
	free(common.endoRow);
	free(common.endoCol);
	free(common.tEndoRowLoc);
	free(common.tEndoColLoc);
		#ifdef TIME_IT
    	aux1T = get_time();
    	#endif
        sycl::free(common.d_endoRow, q_ct1);
        sycl::free(common.d_endoCol, q_ct1);
        sycl::free(common.d_tEndoRowLoc, q_ct1);
        sycl::free(common.d_tEndoColLoc, q_ct1);

        sycl::free(common.d_endoT, q_ct1);
		#ifdef TIME_IT
    	aux2T = get_time();
		freeT += aux2T-aux1T;
    	#endif
        // epi points
	free(common.epiRow);
	free(common.epiCol);
	free(common.tEpiRowLoc);
	free(common.tEpiColLoc);
		#ifdef TIME_IT
    	aux1T = get_time();
    	#endif
        sycl::free(common.d_epiRow, q_ct1);
        sycl::free(common.d_epiCol, q_ct1);
        sycl::free(common.d_tEpiRowLoc, q_ct1);
        sycl::free(common.d_tEpiColLoc, q_ct1);

        sycl::free(common.d_epiT, q_ct1);
		#ifdef TIME_IT
    	aux2T = get_time();
		freeT += aux2T-aux1T;
    	#endif
        //====================================================================================================
	//	POINTERS
	//====================================================================================================
	#ifdef TIME_IT
    aux1T = get_time();
    #endif
	for(i=0; i<common.allPoints; i++){
                sycl::free(unique[i].d_in2, q_ct1);

                sycl::free(unique[i].d_conv, q_ct1);
                sycl::free(unique[i].d_in2_pad_cumv, q_ct1);
                sycl::free(unique[i].d_in2_pad_cumv_sel, q_ct1);
                sycl::free(unique[i].d_in2_sub_cumh, q_ct1);
                sycl::free(unique[i].d_in2_sub_cumh_sel, q_ct1);
                sycl::free(unique[i].d_in2_sub2, q_ct1);
                sycl::free(unique[i].d_in2_sqr, q_ct1);
                sycl::free(unique[i].d_in2_sqr_sub2, q_ct1);
                sycl::free(unique[i].d_in_sqr, q_ct1);

                sycl::free(unique[i].d_tMask, q_ct1);
                sycl::free(unique[i].d_mask_conv, q_ct1);
    }
	#ifdef TIME_IT
    aux2T = get_time();
	freeT += aux2T-aux1T;
    #endif

	#ifdef TIME_IT
	long long totalTime = initT+alocT+cpinT+kernT+cpouT+freeT;
	printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

	printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",	(float) initT / 1000000, (float) initT / (float) totalTime * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 					(float) alocT / 1000000, (float) alocT / (float) totalTime * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",					(float) cpinT / 1000000, (float) cpinT / (float) totalTime * 100);

	printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) kernT / 1000000, (float) kernT / (float) totalTime * 100);

	printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",				(float) cpouT / 1000000, (float) cpouT / (float) totalTime * 100);
	printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 					(float) freeT / 1000000, (float) freeT / (float) totalTime * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) totalTime / 1000000);
	#endif
}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
