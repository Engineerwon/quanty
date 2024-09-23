import pdb
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)




def get_pile(nsamples, seed, seqlen, model):
    print("get_pile")
    traindata = load_dataset("json", data_files='/cpfs01/user/chenmengzhao/prompt_quantization/val.jsonl.zst', split="train")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_wikitext2(nsamples, seed, seqlen, model):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    print("get_ptb")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    print("get_c4")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    print("get_ptb_new")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata  = load_dataset('ptb_text_only', 'penn_treebank', split='test')


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata ["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model): #nsamples는 훈련 데이터에서 추출할 샘플 수, seed는 랜덤 시드, seqlen은 각 입력 텍스트의 최대 시퀀스 길이.
    print("get_c4_new")

    #이건 논문 저자가 허깅페이스 상에서 데이터셋을 마련해 놓았기 때문에 이렇게 불러와서 쓰는 것... 
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False) #빠른 토크나이저를 쓰지 않는 거라는데 잘 모르겠다.
    
    random.seed(seed) #주어진 시드를 이용해서 랜덤 샘플링이 항상 동일한 결과가 나오도록 함.
    trainloader = [] #리스트
    for _ in range(nsamples):
        while True: #계속 루프 돌려라.
            i = random.randint(0, len(traindata) - 1) #traindata에서 랜덤으로 하나를 고른다. 숫자로.
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt") #파이토치 텐서 타입의 traindata에서 i번째 키의 datadict의 "text" key에 있는 줄글을 부름.
            if trainenc.input_ids.shape[1] >= seqlen: #input_ids키에 들어있는 텐서는 (bsz, seqlen) 모양의 텐서이다. 당연히 break아님?
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1) #텍스트의 무작위 시작지점 지정. seqlen만큼 슬라이스하려면 시작점이 맨 끝에서 seqlen개 더 앞에 있어야함.
        j = i + seqlen #선택한 시작지점에서 seqlen만큼 더해서 끝점 지정
        inp = trainenc.input_ids[:, i:j] #두번째 차원, i부터 j를 슬라이싱 이걸 inp로 지정
        tar = inp.clone() #inpt를 복사해서 tar(목표)로 지정. 다음 단어를 예측하도록 만드는 것. 입력과 모양은 똑같아야 된다.
        tar[:, :-1] = -100 # 목표텐서의 마지막 부분을 제외하고 다 -100으로 채움. -100은 pytorch의 cross entropy에서 무시할 값으로 지정됨. 이것으로 마지막 단어만 예측하도록 만드는 것임.
        trainloader.append((inp, tar))# 입력텐서와 목표 텐서를 튜플로 묶고 trainloader 리스트에 append.
    #검증 데이터 만들기
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt") #valdata의 인덱스 0에서 1099까지 텍스트만 공백을 사이에 두고 다 하나의 문자열로 join. 그리고 이걸 토크나이저!
    valenc = valenc.input_ids[:, : (256 * seqlen)] #두번째 차원의 (256*seqlen-1)번 인덱스까지만 나오도록 슬라이싱
    return trainloader, valenc


# def get_loaders(
#     name, nsamples=128, seed=0, seqlen=2048, model='',
# ):
#     if 'wikitext2' in name:
#         return get_wikitext2(nsamples, seed, seqlen, model)
#     if 'pile' in name:
#         return get_pile(nsamples, seed, seqlen, model)
#     if 'ptb' in name:
#         if 'new' in name:
#             return get_ptb_new(nsamples, seed, seqlen, model)  
#         return get_ptb(nsamples, seed, seqlen, model)
#     if 'c4' in name:
#         if 'new' in name:
#             return get_c4_new(nsamples, seed, seqlen, model)  
#         return get_c4(nsamples, seed, seqlen, model)
#     if 'mix' in name:
#         wiki_train,wiki_val=get_wikitext2(nsamples//3, seed, seqlen, model)
#         ptb_train,ptb_val=get_ptb(nsamples//3, seed, seqlen, model)
#         c4_train,c4_val=get_c4(nsamples//3, seed, seqlen, model)
#         train=wiki_train+ptb_train+c4_train
#         val=None
#         return train,val



def get_glue(nsamples, seed, seqlen, model, task_name='mrpc'):
    """
    GLUE 데이터셋을 BERT 모델에 맞게 전처리하여 로드하는 함수
    task_name: 'mrpc', 'sst2', 'qqp', 'mnli' 등 다양한 GLUE 태스크 이름을 사용할 수 있습니다.
    """
    print(f"get_glue - task: {task_name}")
    
    # GLUE 데이터셋 로드
    dataset = load_dataset('glue', task_name)
    traindata = dataset['train']
    valdata = dataset['validation']

    # BERT용 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # 데이터 전처리 함수
    def encode(examples):
        if task_name in ['mrpc', 'qqp', 'mnli']:  # 문장 쌍을 처리하는 태스크
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=seqlen)
        elif task_name == 'sst2':  # 단일 문장을 처리하는 태스크
            return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=seqlen)
        else:
            raise ValueError(f"Unsupported task: {task_name}")

    # 훈련 데이터와 검증 데이터 토크나이즈
    train_encodings = traindata.map(encode, batched=True)
    val_encodings = valdata.map(encode, batched=True)

    # PyTorch Tensor로 변환
    random.seed(seed)
    trainloader = []
    for i in range(min(nsamples, len(train_encodings))):
        input_ids = torch.tensor(train_encodings[i]['input_ids'])
        attention_mask = torch.tensor(train_encodings[i]['attention_mask'])
        token_type_ids = torch.tensor(train_encodings[i].get('token_type_ids', []))  # 일부 태스크에서 token_type_ids가 없을 수 있음
        label = torch.tensor(train_encodings[i]['label'])
        trainloader.append((input_ids, attention_mask, token_type_ids, label))

    valloader = []
    for i in range(min(256, len(val_encodings))):
        input_ids = torch.tensor(val_encodings[i]['input_ids'])
        attention_mask = torch.tensor(val_encodings[i]['attention_mask'])
        token_type_ids = torch.tensor(val_encodings[i].get('token_type_ids', []))  # 일부 태스크에서 token_type_ids가 없을 수 있음
        label = torch.tensor(val_encodings[i]['label'])
        valloader.append((input_ids, attention_mask, token_type_ids, label))

    return trainloader, valloader



#bert까지 고려해서 gpt한테 짜달라고 함.
def get_loaders(name, nsamples=128, seed=0, seqlen=512, model='', task_name='mrpc'):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'pile' in name:
        return get_pile(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)  
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)  
        return get_c4(nsamples, seed, seqlen, model)
    if 'sst2' in name:
        return get_glue(nsamples, seed, seqlen, model, task_name='sst2')
    if 'mrpc' in name:
        return get_glue(nsamples, seed, seqlen, model, task_name='mrpc')
    if 'qqp' in name:
        return get_glue(nsamples, seed, seqlen, model, task_name='qqp')
    
    if 'mix' in name:
        wiki_train, wiki_val = get_wikitext2(nsamples // 3, seed, seqlen, model)
        ptb_train, ptb_val = get_ptb(nsamples // 3, seed, seqlen, model)
        c4_train, c4_val = get_c4(nsamples // 3, seed, seqlen, model)
        train = wiki_train + ptb_train + c4_train
        val = None
        return train, val
