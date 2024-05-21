#!/bin/bash


datadir=$HOME/datain
cmd="src/bntree/data2forest.py $datadir/$1/$1.vd $datadir/$1/$1.idt $1.bn"  
echo $cmd
