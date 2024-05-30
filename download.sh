#!/bin/bash

mkdir -p datasets
cd datasets

for dataset_name in "arguana" "climate-fever" "cqadupstack" "fiqa" "nq" "dbpedia-entity" "fever" "msmarco" "hotpotqa" "nfcorpus" "quora" "scidocs" "scifact" "webis-touche2020" "trec-covid"; do
    wget "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${dataset_name}.zip"
    unzip "${dataset_name}.zip"
done