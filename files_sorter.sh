#! /bin/bash

for energy in {10,25,50,75,100,125,150,175}; do
    for fID in {0..34}; do
        echo "data/${energy}GeV/converted_${energy}GeV_debug_$fID.root" >> photons_train.txt
    done
    for fID in {35..49}; do
        echo "data/${energy}GeV/converted_${energy}GeV_debug_$fID.root" >> photons_test.txt
    done
done