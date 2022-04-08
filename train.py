import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"       ### only use two GPUs
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sentence_process import cut_sentence
from SegmentBERT import SegmentBERT
import random

punctuation_ids = {'，': 8024, '。': 511, '（': 8020, '）': 8021, '《': 517, '》': 518, '"': 107, '\'':112, '！': 8013, '、': 510, '℃': 360, '##℃': 8320, '：': 8038, '；': 8039, '？': 8043, '…': 8106, '●': 474, '／': 8027, '①': 405, '②': 406, '③': 407, '④': 408, '⑤': 409, '⑥': 410, '⑦': 411, '⑧': 412, '⑨': 413, '⑩': 414, '＊': 8022, '〈': 515, '〉': 516, '『': 521, '』': 522, '＇': 8019, '｀': 8050, '.': 119, '「': 519, '」': 520}

def dist(x, y):
    '''
    distance function
    '''
    return np.sqrt(((x - y)**2).sum())

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese-pytorch_model/vocab.txt')
bert_config = BertConfig.from_json_file('bert-base-chinese-pytorch_model/bert_config.json')

device = 0

model = SegmentBERT(bert_config)
state_dict = model.state_dict()
state_dict_pretrained = torch.load('bert-base-chinese-pytorch_model/bert-base-chinese-pytorch_model.bin', map_location='cpu')
state_dict_pretrained = {k.replace('.gamma','.weight').replace('.beta','.bias'):v for k,v in state_dict_pretrained.items()}
for n, p in enumerate(state_dict.keys()):
    if n >= 205:
        break
    state_dict[p] = state_dict_pretrained[p]

model.load_state_dict(state_dict)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.to(device=device)

dataset = 'pku' # 'pku' or 'msr'
output_dir = './saved_models'

texts = []
with open(f'dataset/testing/{dataset}_test.utf8', 'r') as f:
    for line in f.readlines():
        texts.extend(cut_sentence(line))
texts = [i for i in texts if len(i) > 1]
num_sample = len(texts)

def train1(model, start=0, end=0, save_model=True):
    '''
    discriminative module learns from generative module
    '''
    optimizer.zero_grad()
    masked_lm_loss = 0.0
    discriminative_loss = 0.0
    for epoch in tqdm(range(start, end), unit="epoch", mininterval=10):
        sentence = texts[epoch % num_sample]
        input_ids = tokenizer.encode(sentence, add_special_tokens=False)
        length = len(input_ids)
        input_ids.insert(0, 101)
        input_ids.insert(length + 1, 102)

        # random mask to count masked_lm_loss
        masked_lm_input_ids = torch.tensor(input_ids)
        masked_lm_labels = torch.tensor(input_ids)
        for i in range(1, length + 1):
            r = random.random()
            if r < 0.12:
                masked_lm_input_ids[i] = 103 # id of [mask]
        masked_lm_loss += (model(input_ids=masked_lm_input_ids.unsqueeze(0).to(device), masked_lm_labels=masked_lm_labels.unsqueeze(0).to(device)))[0].cpu()

        # count discriminative_loss
        logits = (model(input_ids=torch.tensor(input_ids).unsqueeze(0).to(device), mode=1))[0].cpu()

        with torch.no_grad():
            ninput_ids = np.array([input_ids] * (2 * length - 1))

            for i in range(length):
                if i > 0:
                    ninput_ids[2 * i - 1, i] = 103 # id of [mask]
                    ninput_ids[2 * i - 1, i + 1] = 103 # id of [mask]
                ninput_ids[2 * i, i + 1] = 103 # id of [mask]

            batch_size = 16
            batch_num = ninput_ids.shape[0] // batch_size if ninput_ids.shape[0] % batch_size == 0 else (ninput_ids.shape[0] // batch_size) + 1
            small_batches = [[ninput_ids[num*batch_size : (num+1)*batch_size]] for num in range(batch_num)]
            for num, [input] in enumerate(small_batches):
                if num == 0:
                    vectors = (model((torch.from_numpy(input)).to(device)))[0].cpu().detach().numpy()
                else:
                    tmp_vectors = (model((torch.from_numpy(input)).to(device)))[0].cpu().detach().numpy()
                    vectors = np.concatenate((vectors, tmp_vectors), axis=0)

            labels = []
            labels.append(0) # [CLS]
            labels.append(0) # first character
            for i in range(1, length): # decide whether the i-th character and the (i+1)-th character should be in one word
                if input_ids[i] in punctuation_ids.values() or input_ids[i + 1] in punctuation_ids.values():
                    labels.append(0)
                    continue
                d1 = dist(vectors[2 * i, i + 1], vectors[2 * i - 1, i + 1])
                d2 = dist(vectors[2 * i - 2, i], vectors[2 * i - 1, i])
                d = (d1 + d2) / 2
                if d >= 12:
                    labels.append(1)
                elif d >= 8:
                    labels.append(-100) # -100 is ignored in CrossEntropyLoss()
                else:
                    labels.append(0)
            labels.append(0) # [SEP]

        loss_fct = CrossEntropyLoss()
        discriminative_loss += loss_fct(logits.view(-1, 2), torch.tensor(labels).view(-1))

        # count total loss and backward
        if ((epoch + 1) % 32 == 0) or (epoch == end - 1):
            k_m = 0.30
            k_d = 1 - k_m
            loss = k_m * masked_lm_loss + k_d * discriminative_loss
            print("epoch {}:\n\t{} * masked_lm_loss = {}\n\t{} * discriminative_loss = {}\n\ttotal loss = {}".format(epoch + 1, k_m, k_m * masked_lm_loss, k_d, k_d * discriminative_loss, loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            masked_lm_loss = 0.0
            discriminative_loss = 0.0

    # save model
    if save_model:
        coreModel = model.module if hasattr(model, "module") else model
        state_dict = coreModel.state_dict()
        torch.save(state_dict, os.path.join(output_dir, f"SegmentBERT_{dataset}_{end // (learning_epoch // 2)}.pkl"))

def train2(model, start=0, end=0, save_model=True):
    '''
    generative module learns from discriminative module
    '''
    optimizer.zero_grad()
    masked_lm_loss = 0.0
    generative_loss = 0.0
    for epoch in tqdm(range(start, end), unit="epoch", mininterval=10):
        sentence = texts[epoch % num_sample]
        input_ids = tokenizer.encode(sentence, add_special_tokens=False)
        length = len(input_ids)
        input_ids.insert(0, 101)
        input_ids.insert(length + 1, 102)

        # random mask to count masked_lm_loss
        masked_lm_input_ids = torch.tensor(input_ids)
        masked_lm_labels = torch.tensor(input_ids)
        for i in range(1, length + 1):
            r = random.random()
            if r < 0.12:
                masked_lm_input_ids[i] = 103 # id of [mask]
        masked_lm_loss += (model(input_ids=masked_lm_input_ids.unsqueeze(0).to(device), masked_lm_labels=masked_lm_labels.unsqueeze(0).to(device)))[0].cpu()

        # count generative_loss
        with torch.no_grad():
            logits = (model(input_ids=torch.tensor(input_ids).unsqueeze(0).to(device), mode=1))[0][0].cpu()
            probs = F.softmax(logits, dim=1)[1:length + 1]
            del logits
            probs = probs.t()
            prob_score = probs[0] - probs[1] # P(label 0) - P(label 1)
            del probs

        ninput_ids = np.array([input_ids] * (2 * length - 1))

        for i in range(length):
            if i > 0:
                ninput_ids[2 * i - 1, i] = 103 # id of [mask]
                ninput_ids[2 * i - 1, i + 1] = 103 # id of [mask]
            ninput_ids[2 * i, i + 1] = 103 # id of [mask]

        batch_size = 16
        batch_num = ninput_ids.shape[0] // batch_size if ninput_ids.shape[0] % batch_size == 0 else (ninput_ids.shape[0] // batch_size) + 1
        small_batches = [[ninput_ids[num*batch_size : (num+1)*batch_size]] for num in range(batch_num)]
        for num, [input] in enumerate(small_batches):
            if num == 0:
                vectors = (model((torch.from_numpy(input)).to(device)))[0].cpu().detach().numpy()
            else:
                tmp_vectors = (model((torch.from_numpy(input)).to(device)))[0].cpu().detach().numpy()
                vectors = np.concatenate((vectors, tmp_vectors), axis=0)

        d_lst = []
        target_lst = []

        for i in range(1, length):
            d1 = dist(vectors[2 * i, i + 1], vectors[2 * i - 1, i + 1])
            d2 = dist(vectors[2 * i - 2, i], vectors[2 * i - 1, i])
            d = (d1 + d2) / 2
            d_lst.append(d)
            if prob_score[i] >= 0.5 and d > 8.0:
                target_lst.append(8.0)
            elif prob_score[i] <= -0.5 and d < 12.0:
                target_lst.append(12.0)
            else:
                target_lst.append(d)

        loss_fct = MSELoss()
        generative_loss += loss_fct(torch.tensor(d_lst), torch.tensor(target_lst))

        # count total loss and backward
        if ((epoch + 1) % 32 == 0) or (epoch == end - 1):
            k_m = 0.80
            k_g = 1.00 - k_m
            loss = k_m * masked_lm_loss + k_g * generative_loss
            print("epoch {}:\n\t{} * masked_lm_loss = {}\n\t{} * discriminative_loss = {}\n\ttotal loss = {}".format(epoch + 1, k_m, k_m * masked_lm_loss, k_g, k_g * generative_loss, loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            masked_lm_loss = 0.0
            generative_loss = 0.0

    # save model
    if save_model:
        coreModel = model.module if hasattr(model, "module") else model
        state_dict = coreModel.state_dict()
        torch.save(state_dict, os.path.join(output_dir, f"SegmentBERT_{dataset}_{end // (learning_epoch // 2)}.pkl"))

# train discriminative module which is randomly initialized
num_epochs = 2
num_training_steps = num_sample * num_epochs
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler_class = get_constant_schedule_with_warmup
scheduler_args = {'num_warmup_steps':int(0.5*num_training_steps)}
scheduler = scheduler_class(**{'optimizer':optimizer}, **scheduler_args)
train1(model, start=0, end=num_training_steps, save_model=False)
# save model
coreModel = model.module if hasattr(model, "module") else model
state_dict = coreModel.state_dict()
torch.save(state_dict, os.path.join(output_dir, f"SegmentBERT_{dataset}_Discriminative.pkl"))

# iterative training
num_epochs = 10
num_training_steps = num_sample * num_epochs
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler_class = get_linear_schedule_with_warmup
scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}
scheduler = scheduler_class(**{'optimizer':optimizer}, **scheduler_args)
learning_epoch = 3200
n = (num_training_steps // learning_epoch) + 1
for i in range(n):
    print("Iterative training: {} / {}".format(i + 1, n))
    train2(model=model, start=i * learning_epoch, end=i * learning_epoch + learning_epoch // 2)
    train1(model=model, start=i * learning_epoch + learning_epoch // 2, end=(i + 1) * learning_epoch)