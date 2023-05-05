import json
import random
from time import sleep
import logging
import argparse
import csv
import os
import math

import paddle
from paddle.io import DataLoader,TensorDataset

from paddlenlp.transformers import (
    AutoModelForCausalLM,
    GPTLMHeadModel,
    AutoConfig,
    AutoTokenizer,
)

from utils.dataset import *
from utils.model_utils import *

def set_seed(seed):
     paddle.seed(seed)
     random.seed(seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To suppress warnings about parallelism in tokenizers
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/subj/",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="subj",
    )
    parser.add_argument(
        "--llm_dir",
        type=str,
        default="gpt2-xl",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # set seeds
    if args.seed is not None:
        set_seed(args.seed)

    #set device
    paddle.device.set_device("gpu:0")

    #load llm
    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir)
    # set pad token ids for batched inference cus gpt2 does not have one
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm=AutoModelForCausalLM.from_pretrained(args.llm_dir)

    #load dataset
    train_dataset=SUBJDataset(args.data_path,'train')
    eval_dataset=SUBJDataset(args.data_path,'test')
    
    #build topk
    train_dataset.build_test_topk(eval_dataset,100)

    #eval
    llm.eval()

    # Eval!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.eval_batch_size}")

    predictions_1=[]
    # predictions_2=[]
    # predictions=[]

    batch_size=args.eval_batch_size
    num_step=math.ceil(len(eval_dataset)/batch_size)
    index=[i for i in range(len(eval_dataset))]

    for step in range(num_step):
        input_ids=index[step*batch_size : (step+1)*batch_size]
        topk_ids=train_dataset.get_test_topk(input_ids,k=100)
        #get sample id
        sample_ids_1=[]
        sample_ids_2=[]
        for ids in topk_ids:
            sample_ids_1.append(ids[0:8])
            sample_ids_2.append(ids[8:16])
        #get sample id
        # sample_ids=[]
        # for ids in topk_ids:
        #     sample_ids.append(ids[0:16])

        #evaluate
        with paddle.no_grad():
            predictions_1.extend(test_by_LLM(llm,train_dataset,eval_dataset,tokenizer,
                                           input_ids,sample_ids_1,args.eval_batch_size,args.max_length))
            # predictions_2.extend(test_by_LLM(llm,train_dataset,eval_dataset,tokenizer,
            #                                input_ids,sample_ids_2,args.eval_batch_size,args.max_length))
            # predictions.extend(mixin_test_by_LLM(llm,train_dataset,eval_dataset,tokenizer,
            #                                  input_ids,sample_ids,args.eval_batch_size,args.max_length))
    
    acc=metric(eval_dataset,predictions_1)
    
    # logging
    save_results_file = os.path.join(args.output_dir, 'results.csv')
    csv_exists = os.path.isfile(save_results_file)
    with open(save_results_file, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['dataset', 'seed', 'acc_1' , 'acc_2'])
        csvwriter.writerow([args.dataset_name,
                            args.seed,
                            acc,
                            acc])

if __name__=="__main__":
    main()