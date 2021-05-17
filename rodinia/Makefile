all: cuda oneAPI

cuda: B+TREE_CU BACKPROP_CU BFS_CU CFD_CU DWT2D_CU GAUSSIAN_CU HEARTWALL_CU HOTSPOT_CU HOTSPOT3D_CU HUFFMAN_CU LAVAMD_CU LUD_CU MYOCYTE_CU NN_CU NW_CU PARTICLEFILTER_CU PATHFINDER_CU SRAD_CU STREAMCLUSTER_CU

oneAPI: B+TREE BACKPROP BFS CFD DWT2D GAUSSIAN HEARTWALL HOTSPOT HOTSPOT3D HUFFMAN LAVAMD LUD MYOCYTE NN NW PARTICLEFILTER PATHFINDER SRAD STREAMCLUSTER


B+TREE_CU:
	cd cuda/b+tree; make time=1
B+TREE:
	cd oneAPI/b+tree; make time=1

BACKPROP_CU: 
	cd cuda/backprop; make time=1
BACKPROP:
	cd oneAPI/backprop; make time=1

BFS_CU:
	cd cuda/bfs; make time=1
BFS:
	cd oneAPI/bfs; make time=1

CFD_CU:
	cd cuda/cfd; make time=1
CFD:
	cd oneAPI/cfd; make time=1

DWT2D_CU:
	cd cuda/dwt2d; make time=1
DWT2D:
	cd oneAPI/dwt2d; make time=1

GAUSSIAN_CU:
	cd cuda/gaussian; make time=1
GAUSSIAN:
	cd oneAPI/gaussian; make time=1

HEARTWALL_CU:
	cd cuda/heartwall; make time=1
HEARTWALL:
	cd oneAPI/heartwall; make time=1

HOTSPOT_CU:
	cd cuda/hotspot; make time=1
HOTSPOT:
	cd oneAPI/hotspot; make time=1

HOTSPOT3D_CU:
	cd cuda/hotspot3D; make time=1
HOTSPOT3D:
	cd oneAPI/hotspot3D; make time=1

HUFFMAN_CU:
	cd cuda/huffman; make time=1
HUFFMAN:
	cd oneAPI/huffman; make time=1

LAVAMD_CU:
	cd cuda/lavaMD; make time=1
LAVAMD:
	cd oneAPI/lavaMD; make time=1

LUD_CU:
	cd cuda/lud; make time=1
LUD:
	cd oneAPI/lud; make time=1

MYOCYTE_CU:
	cd cuda/myocyte; make time=1
MYOCYTE:
	cd oneAPI/myocyte; make time=1

NN_CU:
	cd cuda/nn; make time=1
NN:
	cd oneAPI/nn; make time=1

NW_CU:
	cd cuda/nw; make time=1
NW:
	cd oneAPI/nw; make time=1

PARTICLEFILTER_CU:
	cd cuda/particlefilter; make time=1
PARTICLEFILTER:
	cd oneAPI/particlefilter; make time=1

PATHFINDER_CU:
	cd cuda/pathfinder; make time=1
PATHFINDER:
	cd oneAPI/pathfinder; make time=1

SRAD_CU:
	cd cuda/srad; make time=1
SRAD:
	cd oneAPI/srad; make time=1

STREAMCLUSTER_CU:
	cd cuda/streamcluster; make time=1
STREAMCLUSTER:
	cd oneAPI/streamcluster; make time=1



clean: B+TREE_clean BACKPROP_clean BFS_clean CFD_clean DWT2D_clean GAUSSIAN_clean HEARTWALL_clean HOTSPOT_clean HOTSPOT3D_clean HUFFMAN_clean LAVAMD_clean LUD_clean MYOCYTE_clean NN_clean NW_clean PARTICLEFILTER_clean PATHFINDER_clean SRAD_clean STREAMCLUSTER_clean

B+TREE_clean:
	cd cuda/b+tree; make clean
	cd oneAPI/b+tree; make clean

BACKPROP_clean: 
	cd cuda/backprop; make clean
	cd oneAPI/backprop; make clean

BFS_clean:
	cd cuda/bfs; make clean
	cd oneAPI/bfs; make clean

CFD_clean:
	cd cuda/cfd; make clean
	cd oneAPI/cfd; make clean

DWT2D_clean:
	cd cuda/dwt2d; make clean
	cd oneAPI/dwt2d; make clean

GAUSSIAN_clean:
	cd cuda/gaussian; make clean
	cd oneAPI/gaussian; make clean

HEARTWALL_clean:
	cd cuda/heartwall; make clean
	cd oneAPI/heartwall; make clean

HOTSPOT_clean:
	cd cuda/hotspot; make clean
	cd oneAPI/hotspot; make clean

HOTSPOT3D_clean:
	cd cuda/hotspot3D; make clean
	cd oneAPI/hotspot3D; make clean

HUFFMAN_clean:
	cd cuda/huffman; make clean
	cd oneAPI/huffman; make clean

LAVAMD_clean:
	cd cuda/lavaMD; make clean
	cd oneAPI/lavaMD; make clean

LUD_clean:
	cd cuda/lud; make clean
	cd oneAPI/lud; make clean

MYOCYTE_clean:
	cd cuda/myocyte; make clean
	cd oneAPI/myocyte; make clean

NN_clean:
	cd cuda/nn; make clean
	cd oneAPI/nn; make clean

NW_clean:
	cd cuda/nw; make clean
	cd oneAPI/nw; make clean

PARTICLEFILTER_clean:
	cd cuda/particlefilter; make clean
	cd oneAPI/particlefilter; make clean

PATHFINDER_clean:
	cd cuda/pathfinder; make clean
	cd oneAPI/pathfinder; make clean

SRAD_clean:
	cd cuda/srad; make clean
	cd oneAPI/srad; make clean

STREAMCLUSTER_clean:
	cd cuda/streamcluster; make clean
	cd oneAPI/streamcluster; make clean