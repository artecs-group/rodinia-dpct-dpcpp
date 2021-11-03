all: cuda dpcpp

cuda: B+TREE_CU BACKPROP_CU BFS_CU CFD_CU DWT2D_CU GAUSSIAN_CU HEARTWALL_CU HOTSPOT_CU HOTSPOT3D_CU HUFFMAN_CU LAVAMD_CU LUD_CU MYOCYTE_CU NN_CU NW_CU PARTICLEFILTER_CU PATHFINDER_CU SRAD_CU STREAMCLUSTER_CU

dpcpp: B+TREE BACKPROP BFS CFD DWT2D GAUSSIAN HEARTWALL HOTSPOT HOTSPOT3D HUFFMAN LAVAMD LUD MYOCYTE NN NW PARTICLEFILTER PATHFINDER SRAD STREAMCLUSTER


B+TREE_CU:
	cd cuda/b+tree; make time=1
B+TREE:
	cd dpcpp/b+tree; make time=1

BACKPROP_CU: 
	cd cuda/backprop; make time=1
BACKPROP:
	cd dpcpp/backprop; make time=1

BFS_CU:
	cd cuda/bfs; make time=1
BFS:
	cd dpcpp/bfs; make time=1

CFD_CU:
	cd cuda/cfd; make time=1
CFD:
	cd dpcpp/cfd; make time=1

DWT2D_CU:
	cd cuda/dwt2d; make time=1
DWT2D:
	cd dpcpp/dwt2d; make time=1

GAUSSIAN_CU:
	cd cuda/gaussian; make time=1
GAUSSIAN:
	cd dpcpp/gaussian; make time=1

HEARTWALL_CU:
	cd cuda/heartwall; make time=1
HEARTWALL:
	cd dpcpp/heartwall; make time=1

HOTSPOT_CU:
	cd cuda/hotspot; make time=1
HOTSPOT:
	cd dpcpp/hotspot; make time=1

HOTSPOT3D_CU:
	cd cuda/hotspot3D; make time=1
HOTSPOT3D:
	cd dpcpp/hotspot3D; make time=1

HUFFMAN_CU:
	cd cuda/huffman; make time=1
HUFFMAN:
	cd dpcpp/huffman; make time=1

LAVAMD_CU:
	cd cuda/lavaMD; make time=1
LAVAMD:
	cd dpcpp/lavaMD; make time=1

LUD_CU:
	cd cuda/lud; make time=1
LUD:
	cd dpcpp/lud; make time=1

MYOCYTE_CU:
	cd cuda/myocyte; make time=1
MYOCYTE:
	cd dpcpp/myocyte; make time=1

NN_CU:
	cd cuda/nn; make time=1
NN:
	cd dpcpp/nn; make time=1

NW_CU:
	cd cuda/nw; make time=1
NW:
	cd dpcpp/nw; make time=1

PARTICLEFILTER_CU:
	cd cuda/particlefilter; make time=1
PARTICLEFILTER:
	cd dpcpp/particlefilter; make time=1

PATHFINDER_CU:
	cd cuda/pathfinder; make time=1
PATHFINDER:
	cd dpcpp/pathfinder; make time=1

SRAD_CU:
	cd cuda/srad; make time=1
SRAD:
	cd dpcpp/srad; make time=1

STREAMCLUSTER_CU:
	cd cuda/streamcluster; make time=1
STREAMCLUSTER:
	cd dpcpp/streamcluster; make time=1



clean: B+TREE_clean BACKPROP_clean BFS_clean CFD_clean DWT2D_clean GAUSSIAN_clean HEARTWALL_clean HOTSPOT_clean HOTSPOT3D_clean HUFFMAN_clean LAVAMD_clean LUD_clean MYOCYTE_clean NN_clean NW_clean PARTICLEFILTER_clean PATHFINDER_clean SRAD_clean STREAMCLUSTER_clean

B+TREE_clean:
	cd cuda/b+tree; make clean
	cd dpcpp/b+tree; make clean

BACKPROP_clean: 
	cd cuda/backprop; make clean
	cd dpcpp/backprop; make clean

BFS_clean:
	cd cuda/bfs; make clean
	cd dpcpp/bfs; make clean

CFD_clean:
	cd cuda/cfd; make clean
	cd dpcpp/cfd; make clean

DWT2D_clean:
	cd cuda/dwt2d; make clean
	cd dpcpp/dwt2d; make clean

GAUSSIAN_clean:
	cd cuda/gaussian; make clean
	cd dpcpp/gaussian; make clean

HEARTWALL_clean:
	cd cuda/heartwall; make clean
	cd dpcpp/heartwall; make clean

HOTSPOT_clean:
	cd cuda/hotspot; make clean
	cd dpcpp/hotspot; make clean

HOTSPOT3D_clean:
	cd cuda/hotspot3D; make clean
	cd dpcpp/hotspot3D; make clean

HUFFMAN_clean:
	cd cuda/huffman; make clean
	cd dpcpp/huffman; make clean

LAVAMD_clean:
	cd cuda/lavaMD; make clean
	cd dpcpp/lavaMD; make clean

LUD_clean:
	cd cuda/lud; make clean
	cd dpcpp/lud; make clean

MYOCYTE_clean:
	cd cuda/myocyte; make clean
	cd dpcpp/myocyte; make clean

NN_clean:
	cd cuda/nn; make clean
	cd dpcpp/nn; make clean

NW_clean:
	cd cuda/nw; make clean
	cd dpcpp/nw; make clean

PARTICLEFILTER_clean:
	cd cuda/particlefilter; make clean
	cd dpcpp/particlefilter; make clean

PATHFINDER_clean:
	cd cuda/pathfinder; make clean
	cd dpcpp/pathfinder; make clean

SRAD_clean:
	cd cuda/srad; make clean
	cd dpcpp/srad; make clean

STREAMCLUSTER_clean:
	cd cuda/streamcluster; make clean
	cd dpcpp/streamcluster; make clean