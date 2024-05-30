import numpy as np
import yaml
import diskannpy
import time
import os
import csv
import struct

from sentence_transformers import SentenceTransformer, util
import pickle
import evaluate
import pandas as pd
import heapq
import bisect
import pytrec_eval
from collections import defaultdict

import math

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import sys

from tqdm import tqdm
import random

import json

from datetime import datetime

import copy
from copy import deepcopy
import multiprocessing
from multiprocessing import Pool
import sys
import re

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=None, help="which data set to use")
    parser.add_argument("--algo_name", type=str, default="diskann", help="which algorithm to use")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the cheap biencoder name")
    parser.add_argument("--expensive_model_name", type=str, default=None, help="Name of the expensive biencoder name")
    parser.add_argument("--test_strategy", type=str, default="two", help="which test strategy to use")
    parser.add_argument("--graph_degree", type=int, default=64, help="what max degree to use when building the index")
    return parser.parse_args()

args = parse_arguments()

dataset_name=args.dataset_name

cqa_sublist=["android","english","gaming","gis","mathematica","physics","programmers","stats","tex","unix","webmasters","wordpress"]

test_strategy=args.test_strategy
graph_degree = args.graph_degree
algo_name=args.algo_name

print(dataset_name)

data_path = os.path.join("./datasets/",dataset_name.replace("_","/"))

if "msmarco" in dataset_name:
    split="dev"
else:
    split="test"

if dataset_name=="cqadupstack":
    # merge different sub tasks within cqadupstack
    corpus={}
    queries={}
    qrels={}
    for cqa_dataset_name in cqa_sublist:
        data_path = os.path.join("./datasets/","cqadupstack",cqa_dataset_name)
        current_corpus, current_queries, current_qrels = GenericDataLoader(data_folder=data_path).load(split=split)    
        for pid, value in current_corpus.items():
            corpus[pid+cqa_dataset_name]=value
        for qid, value in current_queries.items():
            queries[qid+cqa_dataset_name]=value
        for qid,res in current_qrels.items():
            new_res={pid+cqa_dataset_name:value for pid,value in res.items()}
            qrels[qid+cqa_dataset_name]=new_res
else:
    # load the data set
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

# We want to use the evaluator provided by beir. We just put a dummy model name here.
model = DRES(models.SentenceBERT("msmarco-distilbert-cos-v5"), batch_size=16)
retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "cos_sim" for cosine similarity

print("corpus len", len(corpus))
print("query len", len(queries))
print("qrel len", len(qrels))

log_path=f"./experiment_log/mteb_{algo_name}_{args.model_name}_{args.expensive_model_name}_log/"
csv_path=f"./results/mteb_{algo_name}_{args.model_name}_{args.expensive_model_name}_csv/"
embedding_path="./embedding_data/"

for directory in [log_path,csv_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

model_name=args.model_name

if args.expensive_model_name is not None:
    expensive_model_name=args.expensive_model_name
else:
    expensive_model_name="None"

distance_metric = "angular"

metric="expensive"

# set a time tag as filename suffix
time_tag="240528"

output_path=os.path.join(log_path,f"{dataset_name}_{algo_name}_{model_name}_{expensive_model_name}_{split}_{time_tag}")
output_file=open(output_path,"w")


csv_name=f"{dataset_name}_{algo_name}_{model_name}_{expensive_model_name}_{split}_deg{args.graph_degree}_{test_strategy}_{time_tag}.csv"

csv_file=open(os.path.join(csv_path,csv_name), 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["algo_name","query_L","query_quota","second_query_L","second_query_quota","ndcg_gt","recall_mistral","# of bi-encoder evals","# of expensive bi-encoder evals","running time"])



freeze_comp_comp=False
default_best=None

passage=[]
passage_id=[]
passage_name_id={}
for pid, value in corpus.items():
    passage_id.append(pid)
    passage_name_id[pid]=len(passage)
    passage.append(value["title"]+" "+value["text"])

print("read passage completes")

passage_embedding_name=f"{dataset_name}_{model_name}_passage_embeddings.npy"
if os.path.exists(os.path.join(embedding_path,passage_embedding_name)):
    passage_embeddings=np.load(os.path.join(embedding_path,passage_embedding_name))
    passage_embeddings=passage_embeddings.astype(np.float32)
    print("load from file")
    print("passage embeddings completes", passage_embeddings.shape)
else:
    print("no bi-encoder embedding provided")
    assert(False)

expensive_passage_embedding_name=f"{dataset_name}_{expensive_model_name}_passage_embeddings.npy"
if os.path.exists(os.path.join(embedding_path,expensive_passage_embedding_name)):
    expensive_passage_embeddings=np.load(os.path.join(embedding_path,expensive_passage_embedding_name))
    expensive_passage_embeddings=expensive_passage_embeddings.astype(np.float32)
    print("load from file")
    print("expensive passage embeddings completes", expensive_passage_embeddings.shape)
else:
    expensive_passage_embeddings=None

query=[]
query_id={}
query_name=[]
for qid, value in queries.items():
    query_id[qid]=len(query)
    query_name.append(qid)
    query.append(value)
print("read query completes")

query_embedding_name=f"{dataset_name}_{split}_{model_name}_query_embeddings.npy"
print(query_embedding_name)
if os.path.exists(os.path.join(embedding_path,query_embedding_name)):
    query_embeddings = np.load(os.path.join(embedding_path,query_embedding_name))
    query_embeddings=query_embeddings.astype(np.float32)
    print("load from file")
else:
    print("no bi-encoder embedding provided")
    assert(False)
print("query embeddings complete", query_embeddings.shape)

expensive_query_embedding_name=f"{dataset_name}_{split}_{expensive_model_name}_query_embeddings.npy"
if os.path.exists(os.path.join(embedding_path,expensive_query_embedding_name)):
    expensive_query_embeddings=np.load(os.path.join(embedding_path,expensive_query_embedding_name))
    expensive_query_embeddings=expensive_query_embeddings.astype(np.float32)
    print("load from file")
    print("expensive query embeddings completes", expensive_query_embeddings.shape)
else:
    expensive_query_embeddings=None


k = 10
if algo_name=="diskann":

    def get_index_query_parameters(config_path):
        configurations = []
        with open(config_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)["float"][distance_metric]
        for algo_type in config:
            if algo_type["name"] != "vamana(diskann)":  # Avoids vamana-pq(diskann)
                continue
            configurations += algo_type["run_groups"].values()
        return configurations

    config_yml_path = "vamana-config.yaml"
    configurations = get_index_query_parameters(config_yml_path)
    configuration=configurations[-1]

    args = configuration["args"][0]
    alpha = args["alpha"]
    complexity=args["l_build"]
    prefix = f"{dataset_name}-{model_name}-{alpha}-{complexity}-{graph_degree}"

    print(prefix)

    if not os.path.exists("./indices"):
        os.makedirs("./indices")
    directory = "./indices"

    if not os.path.exists(directory + "/" + prefix):
        diskannpy.build_memory_index(
            passage_embeddings,
            alpha=alpha,
            complexity=complexity,
            graph_degree=graph_degree,
            distance_metric="cosine",
            index_directory=directory,
            num_threads=0,
            use_pq_build=False,
            use_opq=False,
            index_prefix=prefix,
        )

groundtruth={}
for qid, res in qrels.items():
    groundtruth[qid]=[]
    for pid, value in res.items():
        if value!=0:
            groundtruth[qid].append((value,passage_name_id[pid]))
    groundtruth[qid]=[x[1] for x in sorted(groundtruth[qid],reverse=True)]

comp_count={}
visited_passages=[0]*passage_embeddings.shape[0]

def dist(qid,pid,metric="cos",comp_count_factor=1):
    global comp_count

    if metric=="expensive":
        if expensive_passage_embeddings is not None and expensive_query_embeddings is not None:
            comp_count["expensive"]+=comp_count_factor
            score=util.cos_sim(expensive_query_embeddings[qid],expensive_passage_embeddings[pid])[0,0].item()
            return 1-score
    elif metric=="cos":
        score=util.cos_sim(query_embeddings[qid],passage_embeddings[pid])[0,0].item()
        comp_count["biencoder"]+=comp_count_factor
        return 1-score
    elif metric=="l2":
        l2_dist=np.linalg.norm(query_embeddings[qid]-passage_embeddings[pid],ord=2)
        comp_count["biencoder"]+=comp_count_factor
        return l2_dist

def dist_all(qid,pids,metric="cos",comp_count_factor=1):

    if pids==[]:
        return []

    scores=[]
    for pid in pids:
        scores.append(dist(qid,pid,metric=metric,comp_count_factor=comp_count_factor))
    return scores


graph=[]
start=0
max_deg=0
num_nodes=0

def read_diskann_index(index_path):
    with open(index_path, 'rb') as index_path:
        global graph,start,max_deg,num_nodes

        expected_file_size = struct.unpack('Q', index_path.read(struct.calcsize('Q')))[0]
        max_deg = struct.unpack('I', index_path.read(struct.calcsize('I')))[0]
        start = struct.unpack('I', index_path.read(struct.calcsize('I')))[0]
        file_frozen_pts = struct.unpack('Q', index_path.read(struct.calcsize('Q')))[0]
        # print("indexing file size", expected_file_size)
        # print("max deg", max_deg)
        # print("start point", start)
        # print("frozen points", file_frozen_pts)

        bytes_read=struct.calcsize('Q')*2+struct.calcsize('I')*2
        i=0
        while bytes_read!=expected_file_size:
            deg=struct.unpack('I', index_path.read(struct.calcsize('I')))[0]
            graph.append(list(struct.unpack(f'{deg}I', index_path.read(deg * 4))))
            bytes_read+=(deg+1)*struct.calcsize('I')
            i+=1

        num_nodes=len(graph)

        print("read completes")



if algo_name=="diskann":
    print("read diskann index from binary file")
    index_path=f"./indices/{prefix}"
    read_diskann_index(index_path)

def greedy_search(qid,k_neighbors,search_L,start,metric="cos"):
    global graph

    start_time=time.time()

    vis=[]
    in_Q=[False]*num_nodes
    in_vis=[False]*num_nodes

    if isinstance(start,list):
        cur_dist=[]
        for pid in start:
            cur_dist.append(dist(qid,pid,metric=metric))
        Q=[]
        for i in range(len(start)):
            Q.append((cur_dist[i],start[i]))
            in_Q[start[i]]=True
        heapq.heapify(Q)
    else:
        Q=[(dist(qid,start,metric),start)]
        in_Q[start]=True

    p=Q[0][1]
    while p!=-1:
        if in_vis[p] is False:
            bisect.insort(vis,(Q[0][0],p))
            in_vis[p]=True
        heapq.heappop(Q)

        V=[x for x in graph[p] if in_Q[x] is False]
        cur_dist=dist_all(qid,V,metric)

        for i in range(len(V)):
            x=V[i]
            if in_Q[x] is False:
                heapq.heappush(Q,(cur_dist[i],x))
                in_Q[x]=True

        p=-1
        if len(Q)>0:
            if len(vis)<search_L or Q[0][0]<=vis[search_L-1][0]:
                p=Q[0][1]

    for i in range(min(len(Q),k_neighbors)):
        if in_vis[Q[0][1]]==False:
            bisect.insort(vis,(Q[0][0],Q[0][1]))
        heapq.heappop(Q)

    neighbors=[]
    distances=[]
    for i in range(min(k_neighbors,len(vis))):
        neighbors.append(vis[i][1])
        distances.append(vis[i][0])

    len_Q=len(Q)
    len_V=len(V)

    del vis,in_Q,in_vis,Q

    return neighbors

def greedy_search_quota(qid,k_neighbors,query_quota,start,metric="cos"):
    global graph

    start_time=time.time()

    current_dist_count=0

    vis=[]
    in_Q=[False]*num_nodes
    in_vis=[False]*num_nodes

    sequential_expansion=False

    if isinstance(start,list):
        if len(start)>query_quota:
            start=start[:query_quota]

        cur_dist=dist_all(qid,start,metric=metric)
        current_dist_count+=len(start)

        Q=[]
        for i in range(len(start)):
            Q.append((cur_dist[i],start[i]))
            in_Q[start[i]]=True
        heapq.heapify(Q)
    else:
        Q=[(dist(qid,start,metric),start)]
        current_dist_count+=1
        in_Q[start]=True

    p=Q[0][1]
    p_dist=Q[0][0]
    while current_dist_count<query_quota:
        if in_vis[p]==False:
            bisect.insort(vis,(Q[0][0],p))
            in_vis[p]=True

        heapq.heappop(Q)
        V=[x for x in graph[p] if in_Q[x] is False]
        V=list(set(V))

        if len(V)>query_quota-current_dist_count:
            V=V[:query_quota-current_dist_count]
        cur_dist=dist_all(qid,V,metric)
        current_dist_count+=len(V)
        for i in range(len(V)):
            if in_Q[V[i]]==False:
                heapq.heappush(Q,(cur_dist[i],V[i]))
                in_Q[V[i]]=True

        if len(Q)==0:
            break
        else:
            p=Q[0][1]
            p_dist=Q[0][0]

    for i in range(min(len(Q),k_neighbors)):
        if in_vis[Q[0][1]]==False:
            bisect.insort(vis,(Q[0][0],Q[0][1]))
        heapq.heappop(Q)
        

    neighbors=[]
    distances=[]
    for i in range(min(k_neighbors,len(vis))):
        neighbors.append(vis[i][1])
        distances.append(vis[i][0])

    del vis,in_Q,in_vis,Q

    return neighbors


def rerank(qid, pids, k_neighbors,metric="cos"):

    batch_size=100
    L=[]
    for i in range(0,len(pids),batch_size):
        last=min(i+batch_size,len(pids))
        L+=dist_all(qid,pids[i:last],metric=metric)
    L=zip(L,pids)
    L=sorted(L)

    neighbors=[x[1] for x in L]
    distances=[x[0] for x in L]
    return neighbors[:k_neighbors],distances[:k_neighbors]

if os.path.exists("./search_history") is False:
    os.makedirs("./search_history")

search_results_path=os.path.join("search_history",prefix+"-search.pickle")

if os.path.exists(search_results_path):
    with open(search_results_path, 'rb') as file:
        search_results = pickle.load(file)
else:
    search_results={}

def process_query(qid,retrieval_algo,k_neighbors,query_complexity,query_quota,second_query_complexity,second_query_quota):
    global comp_count,cos_threshold

    comp_count={"biencoder":0,"expensive":0,"processed_query":0}

    comp_count["processed_query"]+=1

    assert(query_complexity==0 or query_quota==0)
    assert(second_query_complexity==0 or second_query_quota==0)

    # if the first stage search is run before, load the results from a global dict "search_results" to save time and space.
    if query_complexity!=0:
        if (qid,query_complexity,"complexity") in search_results:
            retrieval_neighbors=search_results[(qid,query_complexity,"complexity")]
        else:
            retrieval_neighbors=greedy_search(query_id[qid],k_neighbors=query_complexity,search_L=query_complexity,start=start,metric="cos")
    else:
        if (qid,query_quota,"quota") in search_results:
            retrieval_neighbors=search_results[(qid,query_quota,"quota")]
        else:
            retrieval_neighbors=greedy_search_quota(query_id[qid],k_neighbors=query_quota,query_quota=query_quota,start=start,metric="cos")

    if "baseline" in retrieval_algo:
        # bi-metric (baseline)
        second_L=max(k_neighbors,second_query_quota)
        expensive_neighbors,expensive_distances=rerank(query_id[qid],copy.deepcopy(retrieval_neighbors[:second_L]),k_neighbors=k_neighbors,metric="expensive")
        neighbors=expensive_neighbors[:k_neighbors]
        distances=expensive_distances[:k_neighbors]
    elif "ours" in retrieval_algo:
        # bi-metric (ours)
        in_Q=[]
        in_vis=[]

        second_L=min(max(100,int(second_query_quota/2)),len(retrieval_neighbors))
        expensive_neighbors=greedy_search_quota(query_id[qid],k_neighbors=k_neighbors,query_quota=second_query_quota,start=start if second_L==0 else retrieval_neighbors[:second_L],metric="expensive")
        neighbors=expensive_neighbors[:k_neighbors]
        distances=dist_all(query_id[qid],neighbors,metric="expensive",comp_count_factor=0)
    else:
        # only perform a bi-encoder search, no further action is taken
        neighbors=retrieval_neighbors[:k_neighbors]
        distances=dist_all(query_id[qid],neighbors,metric="cos",comp_count_factor=0)

    ret={qid:{}}
    for i in range(len(neighbors)):
        pid=passage_id[neighbors[i]]
        distance=distances[i]
        ret[qid][pid]=1-float(distance)

    return {**ret,**comp_count}

def evaluate_recall(groundtruth_qrels,outputs):
    # calculate recall

    gt_metric="cos" if expensive_model_name=="None" else "expensive"
    print(gt_metric)

    ret=[]
    value_ret=[]
    for qid in groundtruth_qrels:
        assert(qid in outputs)
        assert(len(groundtruth_qrels[qid])==k)
        count=0
        threshold=0
        for pid in groundtruth_qrels[qid]:
            if pid in outputs[qid]:
                count+=1
            threshold=max(threshold,dist(query_id[qid],passage_name_id[pid],metric=gt_metric,comp_count_factor=0))
        ret.append(count/len(groundtruth_qrels[qid]))

    return sum(ret)/len(ret)

def retrieval_test(retrieval_algo="our",k_neighbors=10,query_complexity=0,query_quota=0,second_query_complexity=0,second_query_quota=0):
    global metric,comp_count,in_Q,in_vis,cos_threshold,default_best,openai_scores,visited_passages

    time_count=0

    print("query L", query_complexity)
    print("query L", query_complexity,file=output_file)
    print("query quota", query_quota)
    print("query quota", query_quota,file=output_file)
    print("second query complexity",second_query_complexity)
    print("second query complexity",second_query_complexity,file=output_file)
    print("second query quota",second_query_quota)
    print("second query quota",second_query_quota,file=output_file)

    start_time=time.time()

    predictions_trec={}
    predictions_trec_cr={}
    comp_count={"biencoder":0,"expensive":0,"processed_query":0,"pids":{}}
    omitted=[]
    answer_len=[]
    wrong_stats=[]

    seed_value = 203
    random.seed(seed_value)

    for i in range(passage_embeddings.shape[0]):
        visited_passages[i]=0

    if True:
        # prepare for multiprocessing
        num_processes=32
        
        query_args=[]
        for qid in groundtruth:
            query_args.append((qid,retrieval_algo,k_neighbors,query_complexity,query_quota,second_query_complexity,second_query_quota))
        # assign queries to different processes. maximum 3000 is to prevent memory explosion
        results=[]
        i=0
        while i<len(query_args):
            # prevent memory explosion
            last=min(len(query_args),i+1000)
            with Pool(processes=num_processes) as pool:
                results+=pool.starmap(process_query, query_args[i:last])
            i=last
    else:
        # single process running
        results=[]
        for qid in groundtruth:
            results.append(process_query(qid,retrieval_algo,k_neighbors,query_complexity,query_quota,second_query_complexity,second_query_quota))

    print("all queries completed")
    
    # summarizing results for test ndcg

    predictions_trec={}
    for res in results:
        for key,value in res.items():
            if key in ["biencoder","expensive","processed_query"]:
                comp_count[key]+=value
            else:
                predictions_trec[key]=value

    sample_qid = next(iter(predictions_trec))
    if retrieval_algo=="bi":
        if query_complexity!=0 and (sample_qid,query_complexity,"complexity") in search_results:
            exists_in_search=True
        elif query_quota!=0 and (sample_qid,query_quota,"quota") in search_results:
            exists_in_search=True
        else:
            #new search parameter
            for qid in predictions_trec:
                distances=[]
                for pid,score in predictions_trec[qid].items():
                    distances.append((1-score,passage_name_id[pid]))
                distances=sorted(distances)
                neighbors=[x[1] for x in distances]
                distances=[x[0] for x in distances]
                if query_complexity!=0:
                    search_results[(qid,query_complexity,"complexity")]=neighbors
                else:
                    search_results[(qid,query_quota,"quota")]=neighbors
            with open(search_results_path, 'wb') as file:
                pickle.dump(search_results,file)

    comp_count["biencoder"]/=comp_count["processed_query"]
    comp_count["expensive"]/=comp_count["processed_query"]
    ndcg_score=0


    test_predictions_trec=deepcopy(predictions_trec)
    # standard ndcg score
    ndcg_score_gt, _map, recall_score_gt, precision = retriever.evaluate(qrels, test_predictions_trec, [1,5,10])

    print("Queries:", len(groundtruth),file=output_file)
    print("NDCG: ", ndcg_score_gt,file=output_file)
    # print("Recall: ", recall_score_gt,file=output_file)
    output_file.flush()
    print("Queries:", len(groundtruth))
    print("NDCG: ", ndcg_score_gt)
    # print("Recall: ", recall_score_gt)


    # if you want to also test recall, you need to first run test_strategy="single-metric" to produce groundtruth neighbors
    mistral_qrels_folder="./mistral_qrels"
    if os.path.exists(mistral_qrels_folder) is False:
        os.makedirs(mistral_qrels_folder)
    mistral_qrels_file=os.path.join(mistral_qrels_folder,dataset_name+f"_mistral_qrels.pickle")
    if retrieval_algo=="bi" and model_name=="mistral" and query_quota==gt_query_quota:
        mistral_qrels={}
        for qid in predictions_trec:
            mistral_qrels[qid]={}
            # print(qid,len(predictions_trec[qid]))
            assert(len(predictions_trec[qid])==10)
            for pid in predictions_trec[qid]:
                mistral_qrels[qid][pid]=1
        if os.path.exists(mistral_qrels_file) is False:
            with open(mistral_qrels_file, 'wb') as file:
                pickle.dump(mistral_qrels, file)

    if os.path.exists(mistral_qrels_file):
        with open(mistral_qrels_file, 'rb') as file:
            mistral_qrels=pickle.load(file)

        test_predictions_trec=deepcopy(predictions_trec)
        if k_neighbors==10:
            recall_score_mistral= evaluate_recall(mistral_qrels, test_predictions_trec)
        else:
            recall_score_mistral=0

        print("Queries:", len(groundtruth),file=output_file)
        print("Recall: ", recall_score_mistral,file=output_file)
        output_file.flush()
        print("Queries:", len(groundtruth))
        print("Recall: ", recall_score_mistral)
    else:
        recall_score_mistral=0

    end_time=time.time()
    time_count=end_time-start_time

    print("time used",time_count,file=output_file)
    print("time used",time_count)
    
    csv_writer.writerow([retrieval_algo,query_complexity,query_quota,second_query_complexity,second_query_quota,ndcg_score_gt,recall_score_mistral,comp_count["biencoder"],comp_count["expensive"],time_count])
    csv_file.flush()

# start test

gt_query_quota=100000

if "bi-metric" in test_strategy:
    if dataset_name in ["hotpotqa","msmarco","fever","climate-fever","nq","dbpedia-entity"]:
        first_query_complexity=30000
    else:
        first_query_complexity=5000

    retrieval_test(retrieval_algo="bi",k_neighbors=first_query_complexity,query_complexity=first_query_complexity)

    # bi-metric (ours)
    for second_query_quota in [10,20,30,40,50,60,80,100,120,150,200,250,300,400,500,600,700,800,900,1000,1200,1500,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000,12000,15000,20000,25000,30000]:
        if second_query_quota>first_query_complexity:
            break
        retrieval_test(retrieval_algo="bi(ours)",query_complexity=first_query_complexity,second_query_quota=second_query_quota)

    # bi-metric (baseline)
    for second_query_quota in [10,20,30,40,50,60,80,100,120,150,200,250,300,400,500,600,700,800,900,1000,1200,1500,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000,12000,15000,20000,25000,30000]:
        if second_query_quota>first_query_complexity:
            break
        retrieval_test(retrieval_algo="bi(baseline)",query_complexity=first_query_complexity,second_query_quota=second_query_quota)
elif "single-metric" in test_strategy:
    # single-metric
    # the first query_quota=100000 is to produce groundtruth nearest neighbors for recall test
    for query_quota in [100000,100,120,150,200,250,300,400,500,600,700,800,900,1000,1200,1500,2000,2500,3000,3500,4000,5000,6000,7000,8000,10000,12000,15000,20000,25000,30000]:
        retrieval_test(retrieval_algo="bi",k_neighbors=10,query_quota=query_quota)
