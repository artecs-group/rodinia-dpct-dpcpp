/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>

#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors


typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng,
            sycl::nd_item<3> item_ct1)
{
	//int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
        int globalId =
            item_ct1.get_local_range().get(2) *
                (item_ct1.get_group_range(2) * item_ct1.get_group(1) +
                 item_ct1.get_group(2)) +
            item_ct1.get_local_id(2); // more efficient
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sycl::sqrt((lat - latLong->lat) * (lat - latLong->lat) +
                                  (lng - latLong->lng) * (lng - latLong->lng));
        }
}


long long get_time() {
  struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
  long long initTime;
  long long alocTime = 0;
  long long cpinTime = 0;
  long long kernTime = 0;
  long long cpouTime = 0;
  long long freeTime = 0;
  long long aux1Time;
  long long aux2Time;

  aux1Time = get_time();
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  aux2Time = get_time();
  initTime = aux2Time-aux1Time;

  int i=0;
	float lat, lng;
	int quiet=0,timing=0,platform=0,device=0;

  std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[100];
	int resultsCount=10;

    // parse command line
    if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing, &platform, &device)) {
      printUsage();
      return 0;
    }

    int numRecords = loadData(filename,records,locations);
    if (resultsCount > numRecords) resultsCount = numRecords;

    //for(i=0;i<numRecords;i++)
    //  printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,locations[i].lng);


    //Pointers to host memory
	float *distances;
	//Pointers to device memory
	LatLong *d_locations;
	float *d_distances;


	// Scaling calculations - added by Sam Kauffman
        dpct::device_info deviceProp;
        dpct::dev_mgr::instance().get_device(0).get_device_info(deviceProp);
        dev_ct1.queues_wait_and_throw();
        /*
        DPCT1022:0: There is no exact match between the maxGridSize and the
        max_nd_range size. Verify the correctness of the code.
        */
        unsigned long maxGridX = deviceProp.get_max_nd_range_size()[0];
        unsigned long threadsPerBlock = min(
            deviceProp.get_max_work_group_size(), DEFAULT_THREADS_PER_BLOCK);
        size_t totalDeviceMemory;
	/*
  size_t freeDeviceMemory;
	cudaMemGetInfo(  &freeDeviceMemory, &totalDeviceMemory );
        dev_ct1.queues_wait_and_throw();
        unsigned long usableDeviceMemory = freeDeviceMemory * 85 / 100; // 85% arbitrary throttle to compensate for known CUDA bug
	unsigned long maxThreads = usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
	if ( numRecords > maxThreads )
	{
		fprintf( stderr, "Error: Input too large.\n" );
		exit( 1 );
	}
  */
	unsigned long blocks = ceilDiv( numRecords, threadsPerBlock ); // extra threads will do nothing
	unsigned long gridY = ceilDiv( blocks, maxGridX );
	unsigned long gridX = ceilDiv( blocks, gridY );
	// There will be no more than (gridY - 1) extra blocks
        sycl::range<3> gridDim(1, gridY, gridX);

        if ( DEBUG )
	{
		print( totalDeviceMemory ); // 804454400
		//print( freeDeviceMemory );
		//print( usableDeviceMemory );
		print( maxGridX ); // 65535
                print(deviceProp.get_max_work_group_size()); // 1024
                print( threadsPerBlock );
		//print( maxThreads );
		print( blocks ); // 130933
		print( gridY );
		print( gridX );
	}

	/**
	* Allocate memory on host and device
	*/
	distances = (float *)malloc(sizeof(float) * numRecords);
  aux1Time = get_time();
  d_locations = sycl::malloc_device<LatLong>(numRecords, q_ct1);
  d_distances = sycl::malloc_device<float>(numRecords, q_ct1);
  aux2Time = get_time();
  alocTime += aux2Time-aux1Time;
   /**
    * Transfer data from host to device
    */
  aux1Time = get_time();
  q_ct1.memcpy(d_locations, &locations[0], sizeof(LatLong) * numRecords).wait();
  aux2Time = get_time();
  cpinTime += aux2Time-aux1Time;
    /**
    * Execute kernel
    */
    /*
    DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
  aux1Time = get_time();
    q_ct1.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, threadsPerBlock),
                              sycl::range<3>(1, 1, threadsPerBlock)),
            [=](sycl::nd_item<3> item_ct1) {
                euclid(d_locations, d_distances, numRecords, lat, lng,
                       item_ct1);
            });
    });
    dev_ct1.queues_wait_and_throw();

  aux2Time = get_time();
  kernTime += aux2Time-aux1Time;

    //Copy data from device memory to host memory
  aux1Time = get_time();
    q_ct1.memcpy(distances, d_distances, sizeof(float) * numRecords).wait();
  aux2Time = get_time();
  cpouTime += aux2Time-aux1Time;
        // find the resultsCount least distances
    findLowest(records,distances,numRecords,resultsCount);

    // print out results
    if (!quiet)
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }
    free(distances);
    //Free memory
    aux1Time = get_time();
      sycl::free(d_locations, q_ct1);
      sycl::free(d_distances, q_ct1);
    aux2Time = get_time();
    freeTime += aux2Time-aux1Time;

    if(timing){
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

    }
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
  FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;

  /**Main processing **/

    flist = fopen(filename, "r");
    /*
    if(flist == NULL){
      std::cout << "error opening the file\n";
      exit(0);
    }
    */
	while(!feof(flist)) {
		/**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
//    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d){
    int i;
    if (argc < 2) return 1; // error
    strncpy(filename,argv[1],100);
    char flag;

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'r': // number of results
              i++;
              *r = atoi(argv[i]);
              break;
            case 'l': // lat or lng
              if (argv[i][2]=='a') {//lat
                *lat = atof(argv[i+1]);
              }
              else {//lng
                *lng = atof(argv[i+1]);
              }
              i++;
              break;
            case 'h': // help
              return 1;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
