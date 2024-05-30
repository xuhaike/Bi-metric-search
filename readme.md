# A Bi-metric Framework for Fast Similarity Search

This repository contains to code for our paper xxx.

## Overview

We propose a new "bi-metric" framework for designing nearest neighbor data structures. Our framework assumes two dissimilarity functions: a ground-truth metric that is accurate but expensive to compute, and a proxy metric that is cheaper but less accurate. In both theory and practice, we show how to construct data structures using only the proxy metric such that the query procedure achieves the accuracy of the expensive metric, while only using a limited number of calls to both metrics. In this repository, we apply the framework to the text retrieval problem with two dissimilarity functions evaluated by ML models with vastly different computational costs. We observe that for almost all data sets in the 15 MTEB benchmark, our approach achieves a considerably better accuracy-efficiency tradeoff than the alternatives, such as re-ranking.

## Reproducing our main results

Step 1, you need to generate our environment from "requirement.txt"

```bash
pip install -r requirements.txt
```

Step 2, run "download.sh" to download the data sets you want to test. Currently, we have included all the 15 MTEB retrieval benchmark data sets in the script.

```bash
./download.sh
```

Step 3, run "preprocessing_embeddings.py" to produce all the embeddings for the bi-encoders you want to test (both the expensive model and the model as a distance proxy). Currently, we have included all the 15 MTEB retrieval benchmark data sets and all the 4 bi-encoders in the python code as mentioned in our paper. Note that it takes a bit long to generate embeddings for the "mistral" model. Feel free to adjust the data sets and model names used in the python code.

```bash
./preprocessing_embeddings.py
```

Step 4, run "test.sh" to test different methods on each data set. Currently, we choose "bge-micro" to be the distance proxy model and "mistral" to be the expensive model and we have included all the 15 data sets in the script.

```bash
./test.sh
```

* "bi(ours)" is our method
* "bi(baseline)" is the first retrieve and then rerank baseline method. 
* "single-metric" is to build and search only using the expensive model. 

Note that you need to first run "single-metric" to generate groundtruth nearest neighbors for the purpose of testing recall rate as in the current script.

After that, you should be able to find the csv results in "results" folder. The default script generates the results in Figure 1 of our paper.