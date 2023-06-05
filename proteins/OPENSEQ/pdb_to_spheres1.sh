#!/bin/bash

# inputname is the name of the input .pdb file
# outputname is the name of the output .pos file
# Ni is the number of the first aminoacid
# Nf is the number of the last aminoacid
# BB is the letter corresponding to BB (A, B, C, ...)

inputname=$1
outputname=$2
Ni=$3
Nf=$4
BB=$5

rm $outputname

for i in `seq $Ni $Nf`;
do
    awk 'BEGIN{n=0; mx=0; my=0; mz=0}  /^ATOM/ {if ($6=='$i' && $5=="'$BB'") {aa=$4;mx+=$7;my+=$8;mz+=$9;n++} } END{print '$i', mx/n, my/n, mz/n}' $inputname >> $outputname
done  

