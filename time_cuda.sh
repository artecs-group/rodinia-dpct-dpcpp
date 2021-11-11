#!/bin/bash

numExecutions=3

echo "Timing CUDA"

#b+tree
mkdir -p timing/b+tree
cd cuda/b+tree
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./b+tree.out file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt | grep "GPU: KERNEL" >> ../../timing/b+tree/cuda.txt
done

cd ../..

#backprop
mkdir timing/backprop
cd cuda/backprop
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./backprop 65536 | grep "GPU: KERNEL" >> ../../timing/backprop/cuda.txt
done

cd ../..

#bfs
mkdir timing/bfs
cd cuda/bfs
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./bfs ../../data/bfs/graph1MW_6.txt | grep "GPU: KERNEL" >> ../../timing/bfs/cuda.txt
done

cd ../..

#cfd
mkdir timing/cfd
mkdir timing/cfd/euler3d
mkdir timing/cfd/euler3d_double
cd cuda/cfd
pwd
#euler3d
for (( i=0; i<$numExecutions; i++ ))
do
    ./euler3d ../../data/cfd/missile.domn.0.2M | grep "GPU: KERNEL" >> ../../timing/cfd/euler3d/cuda.txt
done

#euler3d_double
for (( i=0; i<$numExecutions; i++ ))
do
    ./euler3d_double ../../data/cfd/missile.domn.0.2M | grep "GPU: KERNEL" >> ../../timing/cfd/euler3d_double/cuda.txt
done

cd ../..

#dwt2d
mkdir timing/dwt2d
cd cuda/dwt2d
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./dwt2d rgb.bmp -d 1024x1024 -f -5 -l 3 | grep "GPU: KERNEL" >> ../../timing/dwt2d/cuda.txt
done

cd ../..

#gaussian
mkdir timing/gaussian
cd cuda/gaussian
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./gaussian -f ../../data/gaussian/matrix1024.txt -q | grep "GPU: KERNEL" >> ../../timing/gaussian/cuda.txt
done

cd ../..

#heartwall
mkdir timing/heartwall
cd cuda/heartwall
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./heartwall ../../data/heartwall/test.avi 104 | grep "GPU: KERNEL" >> ../../timing/heartwall/cuda.txt
done

cd ../..

#hotspot
mkdir timing/hotspot
cd cuda/hotspot
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./hotspot 512 2 2 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out | grep "GPU: KERNEL" >> ../../timing/hotspot/cuda.txt
done

cd ../..

#hotspot3D
mkdir timing/hotspot3D
cd cuda/hotspot3D
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out | grep "GPU: KERNEL" >> ../../timing/hotspot3D/cuda.txt
done

cd ../..

#huffman
mkdir timing/huffman
cd cuda/huffman
pwd
for (( i=0; i<$numExecutions; i++ ))
do
./pavle ../../data/huffman/test1024_H2.206587175259.in | grep "GPU Encoding" >> ../../timing/huffman/cuda.txt
done

cd ../..

#lavaMD
mkdir timing/lavaMD
cd cuda/lavaMD
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./lavaMD -boxes1d 10 | grep "GPU: KERNEL" >> ../../timing/lavaMD/cuda.txt
done

cd ../..

#lud
mkdir timing/lud
cd cuda/lud
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    cuda/lud_cuda -i ../../data/lud/2048.dat | grep "GPU: KERNEL" >> ../../timing/lud/cuda.txt
done

cd ../..

#myocyte
mkdir timing/myocyte
cd cuda/myocyte
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./myocyte.out 100 1 0 | grep "RUN COMPUTATION" >> ../../timing/myocyte/cuda.txt
done

cd ../..

#nn
mkdir timing/nn
cd cuda/nn
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./nn filelist_4 -r 5 -lat 30 -lng 90 -t | grep "GPU: KERNEL" >> ../../timing/nn/cuda.txt
done

cd ../..

#nw
mkdir timing/nw
cd cuda/nw
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./needle 12000 10 | grep "GPU: KERNEL" >> ../../timing/nw/cuda.txt
done

cd ../..

#particlefilter
mkdir timing/particlefilter
mkdir timing/particlefilter/naive
mkdir timing/particlefilter/float
cd cuda/particlefilter
pwd
#naive 
for (( i=0; i<$numExecutions; i++ ))
do
    ./particlefilter_naive -x 128 -y 128 -z 10 -np 1000000 | grep "TOTAL KERNEL TIME" >> ../../timing/particlefilter/naive/cuda.txt
done

#float 
for (( i=0; i<$numExecutions; i++ ))
do
    ./particlefilter_float -x 128 -y 128 -z 10 -np 1000000 | grep "GPU Execution:" >> ../../timing/particlefilter/float/cuda.txt
done

cd ../..

#pathfinder
mkdir timing/pathfinder
cd cuda/pathfinder
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./pathfinder 100000 1000 20 | grep "GPU: KERNEL" >> ../../timing/pathfinder/cuda.txt
done

cd ../..

#srad
mkdir -p timing/srad
mkdir -p timing/srad/srad_v1
mkdir -p timing/srad/srad_v2
cd cuda/srad/srad_v1
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./srad 100 0.5 512 512 | grep "Total time:" >> ../../../timing/srad/srad_v1/cuda.txt
done

cd ../srad_v2
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./srad 512 512 0 31 0 31 0.5 2 | grep "Total time:" >> ../../../timing/srad/srad_v2/cuda.txt
done

cd ../../..

#streamcluster
mkdir timing/streamcluster
cd cuda/streamcluster
pwd
for (( i=0; i<$numExecutions; i++ ))
do
    ./sc_gpu 10 20 256 65536 65536 1000 none output.txt 1 | grep "time kernel" >> ../../timing/streamcluster/cuda.txt
done

cd ../..
