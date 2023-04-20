import json
import random
from time import sleep
import logging
import argparse
import csv
import os

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
        "--num_train_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
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

    index_dataset=TensorDataset([paddle.to_tensor([i for i in range(len(train_dataset))]).unsqueeze(1).cpu()])

    def combine(data_list):
        index_list=[]
        for i in range(len(data_list)):
            index_list.append(data_list[i][0])
        return paddle.concat(index_list)

    train_dataloader=DataLoader(index_dataset,batch_size=args.train_batch_size,shuffle=False,collate_fn=combine)

    #eval
    llm.eval()

    # Eval!
    logger.info("***** Running finding *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.eval_batch_size}")

    predictions=[]
    ids=[]
    for step,batch in enumerate(train_dataloader):
        #sample sample_num trace for each sample
        input_ids=batch
        #get topk_ids
        #[batch_size,k]
        topk_ids=train_dataset.get_bm25_topk(input_ids.cpu().tolist(),k=100)

        #rank
        with paddle.no_grad():
            top8_ids=rank_by_LLM(llm,train_dataset,tokenizer,
                                input_ids,topk_ids,args.eval_batch_size,args.max_length)
        ids.extend(top8_ids)
    
    #logging ids
    save_id_path=os.path.join(args.data_path,"top8.json")
    with open(save_id_path,'w') as w:
        json.dump(ids,w)

if __name__=="__main__":
    main()