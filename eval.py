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

    #load data embeddings
    data_embeddings=paddle.load(os.path.join(args.data_path,"test.pd"))
    #data normalization
    data_norm=paddle.linalg.norm(data_embeddings,2,axis=1).unsqueeze(1)
    data_embeddings=data_embeddings/data_norm

    #load rl model
    model_dir=os.path.join(args.output_dir, args.dataset_name+f"/{args.seed}") 
    rl_model=GPTLMHeadModel.from_pretrained(model_dir)

    #load dataset
    train_dataset=SUBJDataset(args.data_path,'train')
    eval_dataset=SUBJDataset(args.data_path,'test')

    index_dataset=TensorDataset([data_embeddings.cpu(),paddle.to_tensor([i for i in range(len(eval_dataset))]).unsqueeze(1).cpu()])

    def combine(data_list):
        embedding_list=[]
        index_list=[]
        for i in range(len(data_list)):
            embedding_list.append(data_list[i][0])
            index_list.append(data_list[i][1])
        return (paddle.stack(embedding_list),paddle.concat(index_list))

    eval_dataloader=DataLoader(index_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=combine)
    
    #build topk
    train_dataset.build_test_topk(eval_dataset,100)

    #eval
    llm.eval()
    rl_model.eval()

    # Eval!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.eval_batch_size}")

    predictions=[]
    ids=[]
    for step,batch in enumerate(eval_dataloader):
        #sample sample_num trace for each sample
        input_embeddings,input_ids=batch
        #generate
        topk_ids=train_dataset.get_test_topk(input_ids.cpu().tolist(),k=100)
        sampled_ids=generate(rl_model,input_embeddings,topk_ids,8)
        ids.extend(sampled_ids)
        #evaluate
        with paddle.no_grad():
            predictions.extend(test_by_LLM(llm,train_dataset,eval_dataset,tokenizer,
                                           input_ids,sampled_ids,args.eval_batch_size,args.max_length))
    
    acc=metric(eval_dataset,predictions)
    #logging ids
    save_id_dir=os.path.join(args.output_dir,"sample_ids/"+args.dataset_name+f"/{args.seed}")
    if not os.path.exists(save_id_dir):
        os.makedirs(save_id_dir)
    save_id_path=os.path.join(save_id_dir,'ids2.json')
    with open(save_id_path,'w') as w:
        json.dump(ids,w)

    
    # logging
    save_results_file = os.path.join(args.output_dir, 'results_rl2.csv')
    csv_exists = os.path.isfile(save_results_file)
    with open(save_results_file, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['dataset', 'seed', 'acc'])
        csvwriter.writerow([args.dataset_name,
                            args.seed,
                            acc])

if __name__=="__main__":
    main()