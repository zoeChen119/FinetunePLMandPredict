import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys 
sys.path.append("/workspace/ZoeGPT")
print(sys.path)

import json
import torch
import pandas as pd
from tqdm import tqdm
from utils_zoe import mkdir
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

'''读入数据'''
Data_Path = '/workspace/ZoeGPT/BISAI/[DataFountain]基于通用大模型的知识库问答/Data/test.json'
Labeldict_Path = '/workspace/ZoeGPT/BISAI/[DataFountain]基于通用大模型的知识库问答/Data/label.json'
corpus_df = pd.read_json(Data_Path)

if 'label' in corpus_df:
    idx2label = dict()
    # 准备label_dict
    for i,label in enumerate(set(corpus_df['label'])):
        idx2label[i] = label.strip()
else:
    idx2label = dict()
    label_df = pd.read_json(Labeldict_Path)
    for i,row in label_df.iterrows():
        idx2label[row['idx']] = row['label']
'''定义模型参数'''#'/workspace/ZoeGPT/MODELS/roberta-base'
PRETRAINED_MODEL_NAME = '/workspace/ZoeGPT/BISAI/[DataFountain]基于通用大模型的知识库问答/results/intent_cf/chinese-roberta-wwm-ext/batch-64-epoch-30-9.0:1.0-3e-05/fine_tuned_model'
NUM_LABELS = len(idx2label)
batch_size = 64
device = 'cuda:1'
MAX_LEN = 256
REPEAT_TIME = 2

SAVE_PATH_FILE = f'{os.path.dirname(PRETRAINED_MODEL_NAME)}/{os.path.basename(Data_Path).split(".")[-2]}/{PRETRAINED_MODEL_NAME.split("/")[-1]}/'
mkdir(SAVE_PATH_FILE)

'''PLM执行单分类任务:直接预测'''

# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

class MyTestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'idx': torch.tensor(idx, dtype=torch.int),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

question_data = list(corpus_df['question'])

test_dataset= MyTestDataset(question_data, tokenizer, max_length=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 加载预训练的 PLM 模型
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
model = model.to(device)

def batch_predict(save_path):
    predict_df = corpus_df
    predict_df['category'] = None
    # print("开始验证..")
    model.eval()
    pbar = tqdm(total=100)
    with torch.no_grad():
        for batch_idx,batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            for i in range(len(predictions)):
                data_idx = batch_idx*batch_size+i
                predict_df['category'][data_idx] = idx2label[predictions[i].item()]
            pbar.update(batch_size)
    
    predict_df.to_csv(save_path,sep='\t')    
    return predict_df

def single_predict(save_path):
    predict_df = corpus_df
    predict_df['category'] = None
    # print("开始验证..")
    model.eval()
    for idx,row in tqdm(corpus_df.iterrows()):
        input = tokenizer(row['question'],return_tensors='pt').to(device)
        output = model(**input)
        prediction = torch.argmax(output.logits, dim=1)
        
        predict_df['category'][idx] = idx2label[prediction.item()]
    
    predict_df.to_csv(save_path,sep='\t')    
    return predict_df

if __name__=="__main__":
    for experiment_idx in range(1,REPEAT_TIME+1):
        single_predict(SAVE_PATH_FILE+'predict_result_'+str(experiment_idx)+'.csv')
        print(f'第{experiment_idx}次重复实验.')   

        with open(SAVE_PATH_FILE+'0.txt','w',encoding='utf-8') as f:
            f.write(f'Finetuned model path:{PRETRAINED_MODEL_NAME}\n')
            