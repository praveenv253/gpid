#!/bin/bash

for sysname in "binom" "poiss" "binom_zi" "poiss_zi"; do
	for i in {0..10}; do
		sem -j 8 "python3 non_gauss_gt.py $sysname $i"
	done
done
sem --wait
