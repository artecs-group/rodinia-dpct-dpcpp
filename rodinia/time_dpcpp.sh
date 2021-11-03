#!/bin/bash

numExecutions=3

echo "Timing dpcpp"

#b+tree
mkdir -p timing/b+tree
cd dpcpp/b+tree
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./b+tree.out file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt | grep "GPU: KERNEL" >> ../../timing/b+tree/dpcpp.txt
done

cd ../..

#backprop
mkdir -p timing/backprop
cd dpcpp/backprop
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./backprop 65536 | grep "GPU: KERNEL" >> ../../timing/backprop/dpcpp.txt
done

cd ../..

#bfs
mkdir -p timing/bfs
cd dpcpp/bfs
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./bfs ../../data/bfs/graph1MW_6.txt | grep "GPU: KERNEL" >> ../../timing/bfs/dpcpp.txt
done

cd ../..

#cfd
mkdir -p timing/cfd
mkdir -p timing/cfd/euler3d
mkdir -p timing/cfd/euler3d_double
cd dpcpp/cfd
pwd
#euler3d
for (( i=0; i<$numExecutions; i++ ))
do
    ./euler3d ../../data/cfd/missile.domn.0.2M | grep "GPU: KERNEL" >> ../../timing/cfd/euler3d/dpcpp.txt
done

#euler3d_double
for (( i=0; i<$numExecutions; i++ ))
do
    ./euler3d_double ../../data/cfd/missile.domn.0.2M | grep "GPU: KERNEL" >> ../../timing/cfd/euler3d_double/dpcpp.txt
done

cd ../..

#dwt2d
mkdir -p timing/dwt2d
cd dpcpp/dwt2d
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3 | grep "GPU: KERNEL" >> ../../timing/dwt2d/dpcpp.txt
done

cd ../..

#gaussian
mkdir -p timing/gaussian
cd dpcpp/gaussian
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./gaussianElimination -f ../../data/gaussian/matrix1024.txt -q | grep "GPU: KERNEL" >> ../../timing/gaussian/dpcpp.txt
done

cd ../..

#heartwall
mkdir -p timing/heartwall
cd dpcpp/heartwall
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./heartwall ../../data/heartwall/test.avi 104 | grep "GPU: KERNEL" >> ../../timing/heartwall/dpcpp.txt
done

cd ../..

#hotspot
mkdir -p timing/hotspot
cd dpcpp/hotspot
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./hotspot 512 2 2 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out | grep "GPU: KERNEL" >> ../../timing/hotspot/dpcpp.txt
done

cd ../..

#hotspot3D
mkdir -p timing/hotspot3D
cd dpcpp/hotspot3D
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out | grep "GPU: KERNEL" >> ../../timing/hotspot3D/dpcpp.txt
done

cd ../..

#huffman
mkdir -p timing/huffman
cd dpcpp/huffman
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./pavle ../../data/huffman/test1024_H2.206587175259.in | grep "GPU Encoding time" >> ../../timing/huffman/dpcpp.txt
done

cd ../..

#lavaMD
mkdir -p timing/lavaMD
cd dpcpp/lavaMD
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./lavaMD -boxes1d 10 | grep "GPU: KERNEL" >> ../../timing/lavaMD/dpcpp.txt
done

cd ../..

#lud
mkdir -p timing/lud
cd dpcpp/lud
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    cuda/lud_cuda -i ../../data/lud/2048.dat | grep "GPU: KERNEL" >> ../../timing/lud/dpcpp.txt
done

cd ../..

#myocyte
mkdir -p timing/myocyte
cd dpcpp/myocyte
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./myocyte.out 100 1 0 | grep "RUN COMPUTATION" >> ../../timing/myocyte/dpcpp.txt
done

cd ../..

#nn
mkdir -p timing/nn
cd dpcpp/nn
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./nn filelist.txt -r 5 -lat 30 -lng 90 -t | grep "GPU: KERNEL" >> ../../timing/nn/dpcpp.txt
done

cd ../..

#nw
mkdir -p timing/nw
cd dpcpp/nw
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./needleman_wunsch 16000 10 | grep "GPU: KERNEL" >> ../../timing/nw/dpcpp.txt
done

cd ../..

#particlefilter
mkdir -p timing/particlefilter
mkdir -p timing/particlefilter/naive
mkdir -p timing/particlefilter/float
cd dpcpp/particlefilter
pwd
#naive 
for (( i=0; i<$numExecutions; i++ ))
do
    ./particlefilter_naive -x 128 -y 128 -z 10 -np 1000000 | grep "TOTAL KERNEL TIME" >> ../../timing/particlefilter/naive/dpcpp.txt
done

#float 
for (( i=0; i<$numExecutions; i++ ))
do
    ./particlefilter_float -x 128 -y 128 -z 10 -np 1000000 | grep "GPU Execution:" >> ../../timing/particlefilter/float/dpcpp.txt
done

cd ../..

#pathfinder
mkdir -p timing/pathfinder
cd dpcpp/pathfinder
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./pathfinder 100000 1000 20 | grep "GPU: KERNEL" >> ../../timing/pathfinder/dpcpp.txt
done

cd ../..

#streamcluster
mkdir -p timing/streamcluster
cd dpcpp/streamcluster
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./sc_gpu 10 20 256 65536 65536 1000 none output.txt 1 | grep "time kernel" >> ../../timing/streamcluster/dpcpp.txt
done

cd ../..