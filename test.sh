#!/bin/bash

for dataset_name in "arguana" "climate-fever" "cqadupstack" "fiqa" "nq" "dbpedia-entity" "fever" "msmarco" "hotpotqa" "nfcorpus" "quora" "scidocs" "scifact" "webis-touche2020" "trec-covid"; do
    python bi_metric_search.py --algo_name diskann --dataset_name ${dataset_name} --model_name mistral --expensive_model_name None --test_strategy single-metric
    python bi_metric_search.py --algo_name diskann --dataset_name ${dataset_name} --model_name bge-micro --expensive_model_name mistral --test_strategy bi-metric
done