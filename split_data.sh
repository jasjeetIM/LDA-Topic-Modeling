#!/usr/bin/bash
#Script to divide dataset into smaller directories
#Each dir contains 10000 files (18th dir contains 9998 files)

src_dir=./data/PMC_corpus_lite_train/
tar_dir=./data/
tf=($(ls $src_dir | wc -l))
chunk=10000
sub_dir=PMC_
counter=1
while (($tf > 0))
do
  nm=$tar_dir$sub_dir$counter
  mkdir $nm
  counter=$(($counter+1))
  for file in $(ls $src_dir | head -$chunk)
  do
    mv $src_dir$file $nm
  done
  tf=($(ls $src_dir | wc -l))
done 
  
