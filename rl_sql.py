import json
import random
from time import sleep
import logging
import argparse
import csv
import os
import time,math

from visualdl import LogWriter

import paddle
import paddle.nn as nn
from paddle.io import DataLoader,TensorDataset

from paddlenlp.transformers import (
    AutoModelForCausalLM,
    GPTLMHeadModel,
    AutoConfig,
    AutoTokenizer,
)

from paddle.optimizer import AdamW
from paddle.optimizer.lr import OneCycleLR
from utils.dataset import *
from utils.model_utils import *
from utils.loss import soft_q_loss_with_sparse_rewards

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
        default=2,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--reward_factor",
        type=int,
        default=100,
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
        default=0.06,
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
    data_embeddings=paddle.load(os.path.join(args.data_path,"train.pd"))
    #data normalization
    data_norm=paddle.linalg.norm(data_embeddings,2,axis=1).unsqueeze(1)
    data_embeddings=data_embeddings/data_norm

    data_num=data_embeddings.shape[0]

    #load rl config
    rl_config=AutoConfig.from_pretrained("gpt2-en")
    #edit config
    rl_config.num_hidden_layers=2
    rl_config.vocab_size=data_num
    #load rl model
    rl_model=GPTLMHeadModel(config=rl_config)

    #load data embeddings as token embedding
    paddle.nn.utils.vector_to_parameters(data_embeddings.reshape((1,-1)).squeeze(0), rl_model.get_input_embeddings().parameters(), name=None)

    #get a copy target
    rl_model_target=GPTLMHeadModel(config=rl_config)
    rl_model_target.set_state_dict(rl_model.state_dict())

    #load dataset
    train_dataset=SUBJDataset(args.data_path,'train')

    index_dataset=TensorDataset([paddle.to_tensor([i for i in range(len(train_dataset))]).unsqueeze(1).cpu()])

    def combine(data_list):
        index_list=[]
        for i in range(len(data_list)):
            index_list.append(data_list[i][0])
        return paddle.concat(index_list)

    train_dataloader=DataLoader(index_dataset,batch_size=args.train_batch_size,shuffle=True,collate_fn=combine)

    #eval
    llm.eval()
    rl_model.train()
    rl_model_target.train()
    
    #optimizer
    max_train_steps = args.num_train_epochs * len(train_dataloader)
    lr_scheduler = OneCycleLR(
        max_learning_rate=args.learning_rate,
        phase_pct=args.warmup_ratio,
        end_learning_rate=0.0,
        total_steps=max_train_steps,
        anneal_strategy="linear",
    )

    #optimizer
    no_update = ["word_embeddings"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in rl_model.named_parameters() if not any(nd in n for nd in no_update)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in rl_model.named_parameters() if any(nd in n for nd in no_update)],
            "learning_rate": 0.0,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(learning_rate=lr_scheduler,parameters=optimizer_grouped_parameters)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    log_path=os.path.join(args.output_dir,"log/"+args.dataset_name+f"/{args.seed}")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    #log file
    id_file=open(os.path.join(log_path,"id.jsonl"),'w')
    prob_file=open(os.path.join(log_path,"prob.jsonl"),'w')
    reward_file=open(os.path.join(log_path,"reward.jsonl"),'w')
    sql_log_file=open(os.path.join(log_path,"sql_log.jsonl"),'w')

    writer = LogWriter(os.getenv("VDL_LOG_PATH"))
    completed_steps = 0
    time_log = time.time()
    total_loss=0.0
    last_loss=0.0

    for epoch in range(args.num_train_epochs):
        for step,batch in enumerate(train_dataloader):
            #sample sample_num trace for each sample
            input_ids=paddle.repeat_interleave(batch.unsqueeze(1),repeats=args.sample_num,axis=0)

            #get topk example
            topk_ids=train_dataset.get_bm25_topk(input_ids.squeeze(1).cpu().tolist(),k=100)
            #sample
            logits,actions,action_list,action_probs_list=sample_sql(rl_model,input_ids,topk_ids,8,temperature=args.temperature)
            #log ids and probs
            id_file.write(json.dumps(action_list)+"\n")
            prob_file.write(json.dumps(action_probs_list)+"\n")

            #evaluate
            with paddle.no_grad():
                probs=eval_by_LLM_sparse(llm,train_dataset,tokenizer,input_ids.squeeze(1),action_list,args.eval_batch_size,args.max_length)
                #calucu variance reduced rewards
                rewards=calcu_sparse_reward(probs,args.sample_num) * args.reward_factor
                reward_file.write(json.dumps(rewards.cpu().tolist())+"\n")
                #get target_model logits
                target_logits=get_logits(rl_model_target,input_ids,topk_ids,actions)

            #get loss
            loss,sql_log=soft_q_loss_with_sparse_rewards(logits,target_logits,actions,rewards)
            sql_log_file.write(json.dumps(sql_log)+"\n")

            # optimize
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
            completed_steps += 1
            
            #update target model
            state_dict_s=rl_model.state_dict()
            state_dict_t=rl_model_target.state_dict()
            for key,value in state_dict_t.items():
                state_dict_t[key]=value*0.999 + state_dict_s[key] * 0.001
            rl_model_target.set_state_dict(state_dict_t)

            #log loss
            loss_to_log=float(loss.cpu().numpy())
            writer.add_scalar('loss', loss_to_log, completed_steps)
            writer.add_scalar('lr', lr_scheduler.get_lr(), completed_steps)
            total_loss+=loss_to_log

            if completed_steps % 10 == 0:
                logger.info(f"steps: {completed_steps}/{max_train_steps},"
                            f" epoch: {epoch}/{args.num_train_epochs},"
                            f" lr: {lr_scheduler.get_lr():.2e},"
                            f" loss: {(total_loss-last_loss)/10},"
                            f" efficiency: {10 / (time.time() - time_log):.2f}steps/s")
                time_log = time.time()
                last_loss=total_loss
    id_file.close()
    prob_file.close()
    reward_file.close()
    sql_log_file.close()

    save_dir = os.path.join(args.output_dir, args.dataset_name+f"/{args.seed}") 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #save p
    rl_model_target.save_pretrained(save_dir)

    #Eval!
    #load data embeddings
    data_embeddings=paddle.load(os.path.join(args.data_path,"test.pd"))
    #data normalization
    data_norm=paddle.linalg.norm(data_embeddings,2,axis=1).unsqueeze(1)
    data_embeddings=data_embeddings/data_norm
    #load dataset
    eval_dataset=SUBJDataset(args.data_path,'test')

    index_dataset=TensorDataset([data_embeddings.cpu(),paddle.to_tensor([i for i in range(len(eval_dataset))]).unsqueeze(1).cpu()])

    def eval_combine(data_list):
        embedding_list=[]
        index_list=[]
        for i in range(len(data_list)):
            embedding_list.append(data_list[i][0])
            index_list.append(data_list[i][1])
        return (paddle.stack(embedding_list),paddle.concat(index_list))

    eval_dataloader=DataLoader(index_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=eval_combine)
    
    #build topk
    train_dataset.build_test_topk(eval_dataset,100)

    rl_model_target.eval()

    # Eval!
    logger.info("***** Running evaling *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.eval_batch_size}")

    predictions=[]
    ids=[]
    for step,batch in enumerate(eval_dataloader):
        #sample sample_num trace for each sample
        input_embeddings,input_ids=batch
        #generate
        topk_ids=train_dataset.get_test_topk(input_ids.cpu().tolist(),k=100)
        with paddle.no_grad():
            sampled_ids=generate(rl_model_target,input_embeddings,topk_ids,8)
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
    save_id_path=os.path.join(save_id_dir,'ids.json')
    with open(save_id_path,'w') as w:
        json.dump(ids,w)

    # logging results
    save_results_file = os.path.join(args.output_dir, 'results_rl_sql.csv')
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