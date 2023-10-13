import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys 
sys.path.append("/workspace/ZoeGPT")
print(sys.path)

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from utils_zoe import mkdir

'''读入数据'''
Data_Path = '/workspace/ZoeGPT/BISAI/[DataFountain]基于通用大模型的知识库问答/Data/intent_cf.json'

corpus_df = pd.read_json(Data_Path)

intent_dict = dict()
# 准备label_dict
for i,label in enumerate(set(corpus_df['label'])):
    intent_dict[label.strip()] = i

'''定义模型参数'''#'/workspace/ZoeGPT/MODELS/roberta-base'
PRETRAINED_MODEL_NAME = '/workspace/ZoeGPT/BISAI/[DataFountain]基于通用大模型的知识库问答/results/intent_cf/chinese-roberta-wwm-ext/batch-64-epoch-30-9.0:1.0-3e-05/fine_tuned_model'
NUM_LABELS = len(set(corpus_df['label']))
batch_size = 64
LEARNING_RATE = 3e-5
NUM_EPOCHS = 30
device = 'cuda:1'
MAX_LEN = 256
RANDOM_STATE = 0
REPEAT_TIME = 2

# train-test-val
# TEST_SIZE = [0.2,0.5] # 8:1:1
# DATASET_SPLIT = str((1-TEST_SIZE[0])*10) + ':' + str(TEST_SIZE[0]*10*TEST_SIZE[1]) + ':' + str(TEST_SIZE[0]*10*TEST_SIZE[1])

# train-val
TEST_SIZE = 0.4 # 9:1 5:5 3:7 2:8 1:9
DATASET_SPLIT = str(f'{(1-TEST_SIZE)*10:.1f}') + ':' + str(TEST_SIZE*10)

SAVE_PATH_FILE = f'results/{os.path.basename(Data_Path).split(".")[-2]}/{PRETRAINED_MODEL_NAME.split("/")[-1]}/batch-{batch_size}-epoch-{NUM_EPOCHS}-{DATASET_SPLIT}-{LEARNING_RATE}/'
SAVE_PATH_FIG = f'results/{os.path.basename(Data_Path).split(".")[-2]}/{PRETRAINED_MODEL_NAME.split("/")[-1]}/batch-{batch_size}-epoch-{NUM_EPOCHS}-{DATASET_SPLIT}-{LEARNING_RATE}/'
'''PLM执行单分类任务:先微调再预测'''

# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

class MyDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_length):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

question_data = list(corpus_df['question'])
label_data = [intent_dict[label] for label in corpus_df['label']]
all_dataset= MyDataset(question_data, label_data, tokenizer, max_length=MAX_LEN)
# # train-test-val
# train_data, _data = train_test_split(all_dataset, random_state=RANDOM_STATE, test_size=TEST_SIZE[0])
# test_data, val_data = train_test_split(_data, random_state=RANDOM_STATE, test_size=TEST_SIZE[1])
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# train-val
train_data, val_data = train_test_split(all_dataset, random_state=RANDOM_STATE, test_size=TEST_SIZE)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# 加载预训练的 PLM 模型
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
model = model.to(device)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

def one_experiment(experiment_idx):
    losses = []
    accs = []

    # 训练模型
    for epoch in tqdm(range(NUM_EPOCHS)):
        # 训练阶段
        # print("开始微调..")
        model.train()
        for batch in tqdm(train_loader,leave=False):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask,labels=labels)
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            scheduler.step(loss)

        # 验证阶段
        answers = []
        predictions = []
        
        # print("开始验证..")
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in tqdm(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask,labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                answers.extend(list(labels))
                predictions.extend(list(torch.argmax(outputs.logits, dim=-1)))
            
        avg_loss = total_loss / len(val_loader)
        # print(f'Epoch{epoch}: Avg_loss is {avg_loss:.3f}..')
        
        assert len(predictions) == len(answers)
        count = sum(x == y for x, y in zip(answers, predictions))
        # print(f'Epoch{epoch}: 答对了{count}条。')
        acc = count/len(val_data)
        # print(f'Epoch{epoch}: 正确率为{acc:.3%}')
        scheduler.step(avg_loss)
        losses.append(float(avg_loss))
        accs.append(float(acc))


    '''保存acc和loss'''
    mkdir(SAVE_PATH_FILE)
    with open(f'{SAVE_PATH_FILE}{experiment_idx}.txt','w',encoding='utf-8') as f:
        f.write(f'batch_size:{batch_size}\nNUM_EPOCHS:{NUM_EPOCHS}\nLEARNING_RATE:{LEARNING_RATE}\nMAX_LEN:{MAX_LEN}\nDATASET_SPLIT:{DATASET_SPLIT}\nRANDOM_STATE:{RANDOM_STATE}\n')
        f.write('\nacc:\n')
        f.write('\n'.join(list(map(str,accs))))
        f.write('\nloss:\n')
        f.write('\n'.join(list(map(str,losses))))

    '''绘制loss图像'''
    # 设置字体
    font = FontProperties(fname='/workspace/ZoeGPT/学习/TRADITION/simhei.ttf')

    plt.plot(range(NUM_EPOCHS), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'用户意图分类:{SAVE_PATH_FILE}',font=font)

    for i, loss in enumerate(losses):
        plt.text(i, loss, str(f'{loss:.2f}'), ha='center', va='bottom')# 横坐标，纵坐标，文本内容，文本的水平对齐方式，文本的垂直对齐方式

    plt.savefig(f'{SAVE_PATH_FIG}{experiment_idx}.jpg')
    
    '''保存Finetuned模型'''
    model.save_pretrained(f"{SAVE_PATH_FILE}fine_tuned_model")
    tokenizer.save_pretrained(f"{SAVE_PATH_FILE}fine_tuned_model")
    return acc


if __name__=="__main__":
    sum_acc = 0
    for experiment_idx in range(1,REPEAT_TIME+1):
        acc = one_experiment(experiment_idx)
        print(f'第{experiment_idx}次重复实验，最后一个Epoch的准确率为{acc:.3%}')
        sum_acc += acc
    print(f'\n Average Accuracy:{sum_acc/REPEAT_TIME:.3%}')    
    with open(SAVE_PATH_FILE+'0.txt','w',encoding='utf-8') as f:
        f.write(f'batch_size:{batch_size}\nNUM_EPOCHS:{NUM_EPOCHS}\nLEARNING_RATE:{LEARNING_RATE}\nMAX_LEN:{MAX_LEN}\nDATASET_SPLIT:{DATASET_SPLIT}\nRANDOM_STATE:{RANDOM_STATE}\n')
        f.write(f'\n Average Accuracy:{sum_acc/REPEAT_TIME:.3%}')