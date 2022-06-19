#!/bin/bash
#print the directory and file
  
for file in ./inputs/*
do
if [ -d "$file" ]
then
  echo "$file is directory"
elif [ -f "$file" ]
then
  echo "$file is file"
#   sleep 3
  topas $file
fi
done
