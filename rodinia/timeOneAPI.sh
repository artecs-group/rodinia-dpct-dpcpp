#!/bin/bash

numExecutions=3

echo "Timing oneAPI"

#b+tree
mkdir -p timing/b+tree
cd oneAPI/b+tree
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./b+tree.out file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt | grep "GPU: KERNEL" >> ../../timing/b+tree/oneAPI.txt
done

cd ../..

#backprop
mkdir -p timing/backprop
cd oneAPI/backprop
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./backprop 65536 | grep "GPU: KERNEL" >> ../../timing/backprop/oneAPI.txt
done

cd ../..

#bfs
mkdir -p timing/bfs
cd oneAPI/bfs
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./bfs ../../data/bfs/graph1MW_6.txt | grep "GPU: KERNEL" >> ../../timing/bfs/oneAPI.txt
done

cd ../..

#cfd
mkdir -p timing/cfd
mkdir -p timing/cfd/euler3d
mkdir -p timing/cfd/euler3d_double
cd oneAPI/cfd
pwd
#euler3d
for (( i=0; i<$numExecutions; i++ ))
do
    ./euler3d ../../data/cfd/missile.domn.0.2M | grep "GPU: KERNEL" >> ../../timing/cfd/euler3d/oneAPI.txt
done

#euler3d_double
for (( i=0; i<$numExecutions; i++ ))
do
    ./euler3d_double ../../data/cfd/missile.domn.0.2M | grep "GPU: KERNEL" >> ../../timing/cfd/euler3d_double/oneAPI.txt
done

cd ../..

#dwt2d
mkdir -p timing/dwt2d
cd oneAPI/dwt2d
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3 | grep "GPU: KERNEL" >> ../../timing/dwt2d/oneAPI.txt
done

cd ../..

#gaussian
mkdir -p timing/gaussian
cd oneAPI/gaussian
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./gaussianElimination -f ../../data/gaussian/matrix1024.txt -q | grep "GPU: KERNEL" >> ../../timing/gaussian/oneAPI.txt
done

cd ../..

#heartwall
mkdir -p timing/heartwall
cd oneAPI/heartwall
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./heartwall ../../data/heartwall/test.avi 104 | grep "GPU: KERNEL" >> ../../timing/heartwall/oneAPI.txt
done

cd ../..

#hotspot
mkdir -p timing/hotspot
cd oneAPI/hotspot
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./hotspot 512 2 2 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out | grep "GPU: KERNEL" >> ../../timing/hotspot/oneAPI.txt
done

cd ../..

#hotspot3D
mkdir -p timing/hotspot3D
cd oneAPI/hotspot3D
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out | grep "GPU: KERNEL" >> ../../timing/hotspot3D/oneAPI.txt
done

cd ../..

#huffman
mkdir -p timing/huffman
cd oneAPI/huffman
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./pavle ../../data/huffman/test1024_H2.206587175259.in | grep "GPU Encoding time" >> ../../timing/huffman/oneAPI.txt
done

cd ../..

#lavaMD
mkdir -p timing/lavaMD
cd oneAPI/lavaMD
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./lavaMD -boxes1d 10 | grep "GPU: KERNEL" >> ../../timing/lavaMD/oneAPI.txt
done

cd ../..

#lud
mkdir -p timing/lud
cd oneAPI/lud
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    cuda/lud_cuda -i ../../data/lud/2048.dat | grep "GPU: KERNEL" >> ../../timing/lud/oneAPI.txt
done

cd ../..

#myocyte
mkdir -p timing/myocyte
cd oneAPI/myocyte
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./myocyte.out 100 1 0 | grep "RUN COMPUTATION" >> ../../timing/myocyte/oneAPI.txt
done

cd ../..

#nn
mkdir -p timing/nn
cd oneAPI/nn
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./nn filelist.txt -r 5 -lat 30 -lng 90 -t | grep "GPU: KERNEL" >> ../../timing/nn/oneAPI.txt
done

cd ../..

#nw
mkdir -p timing/nw
cd oneAPI/nw
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./needleman_wunsch 16000 10 | grep "GPU: KERNEL" >> ../../timing/nw/oneAPI.txt
done

cd ../..

#particlefilter
mkdir -p timing/particlefilter
mkdir -p timing/particlefilter/naive
mkdir -p timing/particlefilter/float
cd oneAPI/particlefilter
pwd
#naive 
for (( i=0; i<$numExecutions; i++ ))
do
    ./particlefilter_naive -x 128 -y 128 -z 10 -np 1000000 | grep "TOTAL KERNEL TIME" >> ../../timing/particlefilter/naive/oneAPI.txt
done

#float 
for (( i=0; i<$numExecutions; i++ ))
do
    ./particlefilter_float -x 128 -y 128 -z 10 -np 1000000 | grep "GPU Execution:" >> ../../timing/particlefilter/float/oneAPI.txt
done

cd ../..

#pathfinder
mkdir -p timing/pathfinder
cd oneAPI/pathfinder
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./pathfinder 100000 1000 20 | grep "GPU: KERNEL" >> ../../timing/pathfinder/oneAPI.txt
done

cd ../..

#streamcluster
mkdir -p timing/streamcluster
cd oneAPI/streamcluster
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./sc_gpu 10 20 256 65536 65536 1000 none output.txt 1 | grep "time kernel" >> ../../timing/streamcluster/oneAPI.txt
done

cd ../..