#!/bin/bash

declare -a filters=(
    'Keras-MLP'
    'Keras-LR'
    )

declare -a results=(
    'a5_b5'
    'a8_b2'
    'a99_b01'
    )

declare -a performances=(
    1.0
    0.7
    )

for filter in ${filters[@]}; do
    for result in ${results[@]}; do
        for p in ${performances[@]}; do
            python summarize_results.py -p ${p} --filter '{"tag":"'${filter}'"}' "../results_${result}/" "summary_${filter}_${result}_lt${p//./}"
            cp "summary_${filter}_${result}_lt${p//./}/tag_vs_method_loss_val_heatmap_None.svg" "tag_vs_method_loss_val_heatmap_${filter}_${result}_lt${p//./}.svg"
        done
    done
done
