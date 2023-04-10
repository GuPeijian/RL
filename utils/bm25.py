from rank_bm25 import BM25Okapi

def build_bm25_corpus(train_dataset):
    tokenized_corpus=[]
    for sample in train_dataset.data:
        text=sample["sentence"]
        #label_word=train_dataset.label2verb[sample["label"]]
        tokenized_tokens=text.split(" ")
        #tokenized_tokens.append(label_word)
        tokenized_corpus.append(tokenized_tokens)
    return BM25Okapi(tokenized_corpus)

