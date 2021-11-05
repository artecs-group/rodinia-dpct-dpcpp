all: dpcpp

dpcpp: B+TREE BACKPROP BFS CFD DWT2D GAUSSIAN HEARTWALL HOTSPOT HOTSPOT3D HUFFMAN LAVAMD LUD MYOCYTE NN NW PARTICLEFILTER PATHFINDER SRAD STREAMCLUSTER

DEVICE = NVIDIA_GPU # Available: NVIDIA_GPU, INTEL_GPU

B+TREE:
	cd b+tree; make time=1 DEVICE=$(GPU)

BACKPROP:
	cd backprop; make time=1 DEVICE=$(GPU)

BFS:
	cd bfs; make time=1 DEVICE=$(GPU)

CFD:
	cd cfd; make time=1 DEVICE=$(GPU)

DWT2D:
	cd dwt2d; make time=1 DEVICE=$(GPU)

GAUSSIAN:
	cd gaussian; make time=1 DEVICE=$(GPU)

HEARTWALL:
	cd heartwall; make time=1 DEVICE=$(GPU)

HOTSPOT:
	cd hotspot; make time=1 DEVICE=$(GPU)

HOTSPOT3D:
	cd hotspot3D; make time=1 DEVICE=$(GPU)

HUFFMAN:
	cd huffman; make time=1 DEVICE=$(GPU)

LAVAMD:
	cd lavaMD; make time=1 DEVICE=$(GPU)

LUD:
	cd lud; make time=1 DEVICE=$(GPU)

MYOCYTE:
	cd myocyte; make time=1 DEVICE=$(GPU)

NN:
	cd nn; make time=1 DEVICE=$(GPU)

NW:
	cd nw; make time=1 DEVICE=$(GPU)

PARTICLEFILTER:
	cd particlefilter; make time=1 DEVICE=$(GPU)

PATHFINDER:
	cd pathfinder; make time=1 DEVICE=$(GPU)

SRAD:
	cd srad; make time=1 DEVICE=$(GPU)

STREAMCLUSTER:
	cd streamcluster; make time=1 DEVICE=$(GPU)



clean: B+TREE_clean BACKPROP_clean BFS_clean CFD_clean DWT2D_clean GAUSSIAN_clean HEARTWALL_clean HOTSPOT_clean HOTSPOT3D_clean HUFFMAN_clean LAVAMD_clean LUD_clean MYOCYTE_clean NN_clean NW_clean PARTICLEFILTER_clean PATHFINDER_clean SRAD_clean STREAMCLUSTER_clean

B+TREE_clean:

	cd b+tree; make clean

BACKPROP_clean: 

	cd backprop; make clean

BFS_clean:

	cd bfs; make clean

CFD_clean:

	cd cfd; make clean

DWT2D_clean:

	cd dwt2d; make clean

GAUSSIAN_clean:

	cd gaussian; make clean

HEARTWALL_clean:

	cd heartwall; make clean

HOTSPOT_clean:

	cd hotspot; make clean

HOTSPOT3D_clean:

	cd hotspot3D; make clean

HUFFMAN_clean:

	cd huffman; make clean

LAVAMD_clean:

	cd lavaMD; make clean

LUD_clean:

	cd lud; make clean

MYOCYTE_clean:

	cd myocyte; make clean

NN_clean:

	cd nn; make clean

NW_clean:

	cd nw; make clean

PARTICLEFILTER_clean:

	cd particlefilter; make clean

PATHFINDER_clean:

	cd pathfinder; make clean

SRAD_clean:

	cd srad; make clean

STREAMCLUSTER_clean:

	cd streamcluster; make clean
