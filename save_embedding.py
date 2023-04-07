import paddle
from  paddlenlp.transformers import AutoModel,AutoTokenizer

import math,json,os
from tqdm import tqdm

def generate(model,tokenizer,dataset="sst2",mode="train"):
    #load dataset
    data_file=f"./dataset/{dataset}/{mode}.jsonl"
    with open(data_file,'r') as f:
        all_data=f.readlines()
        # generate sample
    batch_size=128
    num_forward_pass=math.ceil(len(all_data)/batch_size)
    ps=[]

    for i in tqdm(range(num_forward_pass),total=num_forward_pass):
        samples=all_data[i*batch_size:(i+1)*batch_size]
        texts=[]
        for sample in samples:
            text=json.loads(sample.strip())["sentence"]
            texts.append(text)

        tokenized=tokenizer.batch_encode(texts,max_length=128,truncation=True,padding=True,return_tensors='pd',return_attention_mask=True,return_token_type_ids=False)
        input_ids=tokenized["input_ids"]
        attention_mask=tokenized["attention_mask"]
        
        output=model(input_ids=input_ids, attention_mask=attention_mask,return_dict=True)
        p=output.last_hidden_state[:,0].cpu()
        ps.append(p)

    final_p=paddle.concat(ps,axis=0)

    assert final_p.shape[0]==len(all_data)

    paddle.save(final_p,f"./dataset/{dataset}/{mode}.pd")


if __name__=="__main__":
    #load model
    paddle.set_device('gpu:0')
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
    model=AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    dataset_list=os.listdir("./dataset")
    
    for dataset in dataset_list:
        print(f"start generate {dataset}")

        generate(model,tokenizer,dataset,"train")
        generate(model,tokenizer,dataset,"test")
    
    
