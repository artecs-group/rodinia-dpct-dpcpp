#!/bin/bash

numExecutions=3

Replace the makefiles to compile for the ruyman docker
read -p "Replace Makefiles? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    cp -RT -v clang++Makefiles/ ./
fi

#Make all tests for timing
echo "Make"
read -p "Clean? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    make clean
fi

read -p "CUDA? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    make all
    cu=1
else
    make oneAPI
fi


#Time all tests
rm -r timing
mkdir timing

if [ $cu == 1 ]
then
    ./timeCUDA.sh
fi
./timeOneAPI.sh
