import numpy as np
import time
import os

# select the gpu you want to use
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sentence_transformers import SentenceTransformer, util
import pickle

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import os

import random

from angle_emb import AnglE, Prompts

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def preprocess(dataset_name,model_name,do_corpus=True,do_query=True):

    if dataset_name=="msmarco":
        split="dev"
    else:
        split="test"
    split="test"

    #### Provide the data_path where scifact has been downloaded and unzipped
    if dataset_name=="cqadupstack":
        corpus={}
        queries={}
        for subtask_name in cqa_sublist:
            data_path = os.path.join("./datasets/","cqadupstack",subtask_name)
            subtask_corpus, subtask_queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
            for key in subtask_corpus:
                corpus[key+subtask_name]=subtask_corpus[key]
            for key in subtask_queries:
                queries[key+subtask_name]=subtask_queries[key]
    else:
        data_path = os.path.join("./datasets/",dataset_name)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    dataset_file_name=dataset_name.replace("/","_")

    if "bge" in model_name:
        if "large" in model_name:
            biencoder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        elif "small" in model_name:
            biencoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        elif "micro" in model_name:
            biencoder = SentenceTransformer('TaylorAI/bge-micro-v2')
        elif "base" in model_name:
            biencoder = SentenceTransformer('BAAI/bge-base-en-v1.5')
    elif "e5" in model_name:
        if "large" in model_name:
            biencoder = SentenceTransformer("intfloat/e5-large-v2")
        elif "small" in model_name:
            biencoder = SentenceTransformer("intfloat/e5-small-v2")
        elif "base" in model_name:
            biencoder = SentenceTransformer("intfloat/e5-base-v2")
        else:
            assert(False)
    elif "gte" in model_name:
        if "small" in model_name:
            biencoder = SentenceTransformer("thenlper/gte-small")
        elif "base" in model_name:
            biencoder = SentenceTransformer("thenlper/gte-base")
        elif "tiny" in model_name:
            biencoder = SentenceTransformer("TaylorAI/gte-tiny")
        else:
            assert(False)
    elif "mistral" in model_name:
        biencoder = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
    elif "uae" in model_name:
        biencoder = SentenceTransformer("WhereIsAI/UAE-Large-V1")
    else:
        print("unsupported model")
        assert(False)

    if do_corpus is True:    
        passage=[]
        for pid, value in corpus.items():
            passage.append(value["title"]+" "+value["text"])

        passage_embedding_name=f"{dataset_file_name}_{model_name}_passage_embeddings.npy"
        if os.path.exists(os.path.join(save_path,passage_embedding_name)) is False:
            if "mistral" in model_name:
                # adjust batch size for mistral model
                passage_embeddings = biencoder.encode(passage, batch_size=1)
            else:
                # use multiple gpus to encoder passages
                pool = biencoder.start_multi_process_pool()
                passage_embeddings = biencoder.encode_multi_process(passage, pool, batch_size=64)
                biencoder.stop_multi_process_pool(pool)

            np.save(os.path.join(save_path,passage_embedding_name),passage_embeddings)
            print("save to file")

    if do_query is True:
        def get_task_def_by_task_name_and_type(task_name: str) -> str:
            if task_name.lower().startswith('cqadupstack'):
                return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'

            task_name_to_instruct: Dict[str, str] = {
                'ArguAna': 'Given a claim, find documents that refute the claim',
                'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim',
                'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia',
                'FEVER': 'Given a claim, retrieve documents that support or refute the claim',
                'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question',
                'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question',
                'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query',
                'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question',
                'NQ': 'Given a question, retrieve Wikipedia passages that answer the question',
                'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question',
                'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
                'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim',
                'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
                'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
            }

            # add lower case keys to match some beir names
            task_name_to_instruct.update({k.lower(): v for k, v in task_name_to_instruct.items()})
            # other cases where lower case match still doesn't work
            task_name_to_instruct['trec-covid'] = task_name_to_instruct['TRECCOVID']
            task_name_to_instruct['climate-fever'] = task_name_to_instruct['ClimateFEVER']
            task_name_to_instruct['dbpedia-entity'] = task_name_to_instruct['DBPedia']
            task_name_to_instruct['webis-touche2020'] = task_name_to_instruct['Touche2020']
            task_name_to_instruct['fiqa'] = task_name_to_instruct['FiQA2018']
            task_name_to_instruct['quora'] = task_name_to_instruct['QuoraRetrieval']
            return task_name_to_instruct[task_name]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'

        query=[]
        for qid, value in queries.items():
            query.append(value)
        if "mistral" in model_name:
            task_def=get_task_def_by_task_name_and_type(dataset_name)
            query = [get_detailed_instruct(task_def, x) for x in query]
        elif "uae" in model_name:
            query=[Prompts.C.format(text=x) for x in query]

        query_embedding_name=f"{dataset_file_name}_{split}_{model_name}_query_embeddings.npy"
        if os.path.exists(os.path.join(save_path,query_embedding_name)) is False:
            if "mistral" in model_name:
                # adjust batch size for mistral model
                query_embeddings = biencoder.encode(query, batch_size=1)
            else:
                # use multiple gpus to encoder queries
                pool = biencoder.start_multi_process_pool()
                query_embeddings = biencoder.encode_multi_process(query, pool, batch_size=64)
                biencoder.stop_multi_process_pool(pool)
            np.save(os.path.join(save_path,query_embedding_name),query_embeddings)
            print("save to file")

if __name__ == "__main__":
    save_path="./embedding_data/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cqa_sublist=["android","english","gaming","gis","mathematica","physics","programmers","stats","tex","unix","webmasters","wordpress"]


    # include all the data sets which you want to test
    for dataset_name in ["nq","fiqa","fever","trec-covid","scifact","msmarco","scidocs","dbpedia-entity","quora","arguana","nfcorpus","hotpotqa","webis-touche2020","climate-fever","cqadupstack"]:
        # include all the bi-encoder modesl you want to test
        for model_name in ["bge-base","gte-small","bge-micro","mistral"]:
            start_time = time.time()

            print("preprocessing",dataset_name,model_name)
            preprocess(dataset_name,model_name,do_corpus=True,do_query=True)

            end_time=time.time()
            print("time used (hours)",(end_time-start_time)/3600)
