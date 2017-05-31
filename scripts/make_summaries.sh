#!/bin/bash

declare -a filters=(
    'Keras-LR'
    'Keras-MLP'
    )

declare -a results=(
    'a10_b0'
    'a9_b1'
    'a7_b3'
    'a5_b5'
    'a4_b6'
    )

declare -a performances=(
    1.0
#    0.7
    )

for filter in ${filters[@]}; do
    for result in ${results[@]}; do
        for p in ${performances[@]}; do
            python summarize_results.py -p ${p} --filter '{"tag":"'${filter}'"}' "../results_${result}/" "summary_${filter}_${result}_lt${p//./}"
            cp "summary_${filter}_${result}_lt${p//./}/model_vs_mixing_matrix_M_loss_val_heatmap_None.svg" "model_vs_mixing_matrix_M_loss_val_heatmap_${filter}_${result}_lt${p//./}.svg"
        done
    done
done

./merge_heatmaps.sh summary_Keras-LR_a\*lt10
./merge_heatmaps.sh summary_Keras-MLP_a\*lt10
