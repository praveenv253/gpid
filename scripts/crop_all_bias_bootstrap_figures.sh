#!/bin/bash
# Should be run from within the scripts directory

cd ../figures
for m in 10 20; do
	for mode in "bit_of_all" "both_unique" "fully_redundant" "high_synergy" "zero_synergy"; do
		pdfcrop "bias-corr--$mode--$m.pdf"
		pdfcrop "bootstrap-ci--$mode--$m.pdf"
	done
done
