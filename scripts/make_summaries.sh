#!/usr/bin/env sh

python summarize_results.py -p 1.0 --filter '{"tag":"Keras-MLP"}' \
    ../results_a8_b2/ summary_keras-MLP_a8_b2
python summarize_results.py -p 0.7 --filter '{"tag":"Keras-MLP"}' \
    ../results_a8_b2/ summary_keras-MLP_a8_b2_performance_lt07
python summarize_results.py -p 1.0 --filter '{"tag":"Keras-MLP"}' \
    ../results_a5_b5/ summary_keras-MLP_a5_b5
python summarize_results.py -p 0.7 --filter '{"tag":"Keras-MLP"}' \
    ../results_a5_b5/ summary_keras-MLP_a5_b5_performance_lt07

python summarize_results.py -p 1.0 --filter '{"tag":"Keras-LR"}' \
    ../results_a8_b2/ summary_keras-LR_a8_b2
python summarize_results.py -p 0.7 --filter '{"tag":"Keras-LR"}' \
    ../results_a8_b2/ summary_keras-LR_a8_b2_performance_lt07
python summarize_results.py -p 1.0 --filter '{"tag":"Keras-LR"}' \
    ../results_a5_b5/ summary_keras-LR_a5_b5
python summarize_results.py -p 0.7 --filter '{"tag":"Keras-LR"}' \
    ../results_a5_b5/ summary_keras-LR_a5_b5_performance_lt07
