import paddle
import paddle.nn.functional as F
import math
from .template import make_prompt

def generate(model,
             input_embeddings=None,
             len=None):
    """
    generate final example ids for test example
    input:
        model: AR model
        input_embeddings: input example embedding, used in test
        len: generate length
    output:
        output_ids: selected ids for test
    """

    #use argmax to choose the next token

    inputs_embeds=input_embeddings.unsqueeze(1)
    output_ids=[]

    for round in range(len):
        logits=model(inputs_embeds=inputs_embeds,return_dict=True).logits[:,-1,:]

        #argmax
        ids=paddle.argmax(logits,axis=1).unsqueeze(-1)

        #add to input
        new_embeds=model.get_input_embeddings()(ids)
        inputs_embeds=paddle.concat((inputs_embeds,new_embeds),axis=1)

        #save ids
        output_ids.append(ids)
    
    output_ids=paddle.concat(output_ids,axis=1).cpu().tolist()
    
    return output_ids
    
def sample(model,
           input_ids=None,
           topk_ids=None,
           length=None,
           temperature=1.0):
    """
    sample RL traces for training
    input:
        model: AR model
        input_ids: input example index,used in training
        topk_ids: bm25 topk id for each example
        length: generate length
        temperature: sample temperature
    output:
        output_ids: selected ids
        output_probs: each trace's probability
    """
    _input_ids=input_ids
    output_ids=[]
    output_probs=[]
    #prepare mask
    M=1e8
    #index 
    index_ids=paddle.to_tensor([[i] for i in range(_input_ids.shape[0])])

    #mask=F.one_hot(_input_ids.squeeze(1),num_classes=model.config.vocab_size)
    mask=make_mask(topk_ids,model.config.vocab_size)
    
    for round in range(length):
        logits=model(input_ids=_input_ids,return_dict=True).logits[:,-1,:]

        #only consider unmasked token
        logits[mask==1]-=M

        #sample
        probs=F.softmax(logits/temperature,axis=-1)
        sampled_ids=paddle.multinomial(probs,num_samples=1)
        sampled_probs=paddle.gather_nd(probs,index=paddle.concat((index_ids,sampled_ids),axis=1)).unsqueeze(1)
        
        #add to input
        _input_ids=paddle.concat((_input_ids,sampled_ids),axis=1)

        #save
        output_ids.append(sampled_ids)
        output_probs.append(sampled_probs)
    
    output_ids=paddle.concat(output_ids,axis=1).cpu().tolist()
    output_probs=paddle.concat(output_probs,axis=1)
    
    return output_ids,output_probs

def llm_gen(model, prompt, tokenizer, max_context_len):
    inputs = tokenizer.batch_encode(prompt,truncation=True,padding=True,return_tensors='pd',return_attention_mask=True,return_token_type_ids=False)
    if inputs['input_ids'].shape[1] > max_context_len:
        inputs['input_ids'] = inputs['input_ids'][:, -max_context_len:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -max_context_len:]
    with paddle.no_grad():
        logits = model.forward(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'],
                               return_dict=True).logits.detach()
    # the output prob is shifted by -1, so we should use the output at the last input token position
    # gen_logits.shape = [B, 1, 50257]
    gen_logits = logits[:, -1, :]

    return gen_logits

def parse_response(gen_logits, tokenizer, id2verb):
    prob_per_cls = []
    for label_verb in id2verb:
        label_verb_token_id = tokenizer.encode(' ' + label_verb)["input_ids"][-1] # note the space before label word
        prob_per_cls.append(gen_logits[:, label_verb_token_id])
    prob_per_cls=paddle.stack(prob_per_cls,axis=1)
    return prob_per_cls

def flatten(input_texts,labels):
    #flatten [[N]*k] -> [N*k] 
    output_texts=[]
    output_labels=[]
    for i,seq in enumerate(input_texts):
        output_texts.extend(seq)
        output_labels.extend(labels[i])
    return output_texts,input_texts

def eval_by_LLM(model,
                dataset=None,
                tokenizer=None,
                query_ids=None,
                example_ids=None,
                batch_size=None,
                max_length=None):
    """
    eval the selected_ids
    input:
        model: LLM for ICL
        tokenizer: tokenize of LLM
        dataset: current dataset
        query_ids: current ids
        example_ids: selected ids
        batch_size: eval batch_size
        max_length: max_length for context
    output:
        rewards: Loss for each model
    """
    num_query=len(query_ids)
    n_shot=len(example_ids[i])
    #loss fuction
    loss_fct=paddle.nn.CrossEntropyLoss(reduction="none")
    
    #make prompts
    input_texts=[[] for _ in range(n_shot)]
    labels=[[[] for _ in range(n_shot)]]
    for i,query_id in enumerate(query_ids):
        query=make_prompt(dataset,[query_id],"inference")
        label=dataset.label2id[dataset[query_id]["label"]]
        for num_example in range(n_shot):
            #select num_example example for prompt
            prompts=make_prompt(dataset,example_ids[i][:num_example+1],"train")
            input_text=prompts+query
            input_texts[num_example].append(input_text)
            #save label
            labels[num_example].append(label)
    #flatten input_texts make input with similar length together    
    input_texts,labels=flatten(input_texts,labels)
    
    #generate
    num_iteration=math.ceil(num_query*n_shot/batch_size)
    loss=[]
    for i in range(num_iteration):
        batch_input_texts=input_texts[i*batch_size:(i+1)*batch_size]
        gen_logits=llm_gen(model,batch_input_texts,tokenizer,max_length)
        label_logits=parse_response(gen_logits,tokenizer,dataset.id2verb)
        
        with paddle.no_grad():
            batch_label=labels[i*batch_size:(i+1)*batch_size]
            batch_label=paddle.to_tensor(batch_label)
            batch_loss=loss_fct(label_logits,batch_label)
            loss.append(batch_loss)
    # [num_query*n_shot]
    loss=paddle.concat(loss)
    # [num_query,n_shot]
    loss=loss.reshape([n_shot,num_query]).transpose()
    return loss

def test_by_LLM(model,
                train_dataset=None,
                dev_dataset=None,
                tokenizer=None,
                query_ids=None,
                example_ids=None,
                batch_size=None,
                max_length=None):
    """
    eval the selected_ids
    input:
        model: LLM for ICL
        tokenizer: tokenize of LLM
        train_dataset: train dataset
        dev_dataset: dev dataset
        query_ids: current ids
        example_ids: selected ids
        batch_size: eval batch_size
        max_length: max_length for context
    output:
        prediction: prediction for test example
    """
    #make prompts
    input_texts=[]
    for i,query_id in enumerate(query_ids):
        query=make_prompt(dev_dataset,[query_id],"inference")
        prompts=make_prompt(train_dataset,example_ids[i],"train")
        input_text=prompts+query
        input_texts.append(input_text)

    #generate
    num_querys=len(query_ids)
    num_iteration=math.ceil(num_querys/batch_size)
    logits=[]
    with paddle.no_grad():
        for i in range(num_iteration):
            batch_input_texts=input_texts[i*batch_size:(i+1)*batch_size]
            gen_logits=llm_gen(model,batch_input_texts,tokenizer,max_length)
            label_logits=parse_response(gen_logits,tokenizer,dev_dataset.id2verb)
            logits.append(label_logits)
    logits=paddle.concat(logits,axis=0)
    prediction=paddle.argmax(logits,axis=1).cpu().tolist()

    return prediction

def make_mask(topk_ids,num_class):
    batch_size=len(topk_ids)
    k=len(topk_ids[0])
    #initail mask
    mask=paddle.ones((batch_size,num_class),dtype="int32")

    #build index
    index_1=paddle.to_tensor(topk_ids,dtype="int32").unsqueeze(2)

    index_0=[]
    for i in range(batch_size):
        index_0.append([i for _ in range(k)])
    index_0=paddle.to_tensor(index_0,dtype="int32").unsqueeze(2)
    #concat and reshape [batch_size*k, 2]
    index=paddle.concat((index_0,index_1),axis=2).reshape([-1,2])
    
    value=paddle.to_tensor([-1]*(batch_size*k),dtype="int32")

    #build mask
    mask=paddle.scatter_nd_add(mask,index,value)

    return mask

def metric(dataset,predctions):
    num_true=0
    for i in range(len(dataset)):
        label=dataset[i]["label"]
        if dataset.label2id[label]==predctions[i]:
            num_true+=1
    acc=num_true/len(dataset)

    return acc

def calcu_advantage(reward,sample_num,gamma=0.9):
    #first clacu future reward
    time_step=reward.shape[1]
    reward_timestep=paddle.chunk(reward,time_step,axis=1)
    future_reward=0
    final_reward=[]
    for i in range(time_step):
        #reverse calcu
        current_reward=reward_timestep[time_step-i-1]+ gamma * future_reward
        final_reward.appen(current_reward)
        future_reward=current_reward
    #reverse back
    final_reward.reverse()
    final_reward=paddle.concat(final_reward,axis=1)
    #calcu mean for each sample_num
    final_reward=final_reward.reshape((-1,sample_num,time_step))
    reward_mean=paddle.mean(final_reward,axis=1,keepdim=True)
    #calcu advantage
    advantage=final_reward-reward_mean

    return advantage
    
def get_loss(probs,advantanges):
    """
    1/N Sigma_n (Sigma_t (A_tlogp_t))
    """
    assert probs.shape==advantanges.shape
    log_p=paddle.log(probs)
    loss=paddle.mean(paddle.sum(advantanges*log_p,axis=1))

    return loss,log_p