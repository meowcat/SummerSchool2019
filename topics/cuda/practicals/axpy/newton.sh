#!/bin/bash

for i in `seq 8 2 20`
do
  echo $i
  srun ./newton $i | grep ^kernel
done
