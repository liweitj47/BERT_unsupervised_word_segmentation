import torch
from transformers import BertTokenizer, BertConfig
from sentence_process import cut_sentence
from text2label import label2text, text2label
from SegmentBERT import SegmentBERT
from tqdm import tqdm
import os

punctuation_ids = {'，': 8024, '。': 511, '（': 8020, '）': 8021, '《': 517, '》': 518, '"': 107, '\'':112, '！': 8013, '、': 510, '℃': 360, '##℃': 8320, '：': 8038, '；': 8039, '？': 8043, '…': 8106, '●': 474, '／': 8027, '①': 405, '②': 406, '③': 407, '④': 408, '⑤': 409, '⑥': 410, '⑦': 411, '⑧': 412, '⑨': 413, '⑩': 414, '＊': 8022, '〈': 515, '〉': 516, '『': 521, '』': 522, '＇': 8019, '｀': 8050, '.': 119, '「': 519, '」': 520}

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese-pytorch_model/vocab.txt')
bert_config = BertConfig.from_json_file('bert-base-chinese-pytorch_model/bert_config.json')

device = torch.device('cuda:2')

model = SegmentBERT(bert_config)
model.to(device = device)

def seg_seqlab(text, model, device):
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    length = len(input_ids)
    input_ids.insert(0, 101)
    input_ids.insert(length + 1, 102)
    vecs = (model(torch.tensor(input_ids).unsqueeze(0).to(device), mode=1))[0][0].cpu()
    labels = []
    labels.append(0)
    for i in range(2, length + 1):
        if (input_ids[i] in punctuation_ids.values()) or (input_ids[i - 1] in punctuation_ids.values()) :
            labels.append(0)
            continue
        if (vecs[i][0] > vecs[i][1]):
            labels.append(0)
        else:
            labels.append(1)
    result = label2text(text, labels)
    return result

def compare(model, input_ids, i, device):
    length = len(input_ids) - 2
    input_id1 = torch.tensor(input_ids).unsqueeze(0)
    token0 = input_ids[i - 1]
    token1 = input_ids[i]
    token2 = input_ids[i + 1]

    original_lm_label = torch.ones_like(input_id1)
    original_lm_label = original_lm_label.masked_fill(original_lm_label.to(torch.bool), -100)

    input_id1[0][i] = 103 # id of [mask]
    original_lm_label[0][i] = token1

    input_id1[0][i + 1] = 103 # id of [mask]

    right_loss = (model(input_ids=input_id1.to(device), masked_lm_labels=original_lm_label.to(device)))[0].cpu()

    input_id1[0][i + 1] = token2
    input_id1[0][i - 1] = 103 # id of [mask]

    left_loss = (model(input_ids=input_id1.to(device), masked_lm_labels=original_lm_label.to(device)))[0].cpu()

    if right_loss > left_loss:
        return 'right'
    else:
        return 'left'

dataset = 'pku' # 'pku' or 'msr'
model_names = ['SegmentBERT_{}_{}'.format(dataset, i) for i in range(1, 37)]

for model_name in model_names:
    state_dict = torch.load('saved_models/' + model_name +'.pkl', map_location='cpu')
    model.load_state_dict(state_dict)
    with open(f"experiment_result/temp_segmented1.utf8", "w") as fo:
        with open(f'dataset/development_set/{dataset}_dev_unseg.utf8', 'r') as f1:
            seg_text = f1.readlines()
            for line in seg_text:
                seg_sentences = cut_sentence(line)
                final_result = str()
                for seg_sentence in seg_sentences:
                    if len(seg_sentence) > 510:
                        print(seg_sentence, "length > 510, can't be processed by BERT.  ")
                        continue
                    seg_result = seg_seqlab(seg_sentence, model, device=device)
                    final_result += seg_result
                fo.write(final_result)
                fo.write('\n')

    os.system(f"perl dataset/scripts/score dataset/gold/{dataset}_training_words.utf8 dataset/development_set/{dataset}_dev.utf8 experiment_result/temp_segmented1.utf8 > experiment_result/temp_score.utf8")

    with open(f"experiment_result/temp_segmented1.utf8", 'r') as ff:
        text = ff.readlines()
    total = 0
    correct = 0
    for string in tqdm(text, mininterval=10):
        ground_truth_label = text2label(string)
        string = string.replace('  ', '')
        sentences = cut_sentence(string)
        length = 0
        for sentence in sentences:
            length += len(tokenizer.encode(sentence, add_special_tokens=False))
        if length != len(ground_truth_label):
            print("Length assertion failed:", string)
            continue
        p = 0
        for sentence in sentences:
            input_ids = tokenizer.encode(sentence, add_special_tokens=False)
            length = len(input_ids)
            input_ids.insert(0, 101)
            input_ids.insert(length + 1, 102)
            labels = ground_truth_label[p : p + length]
            p += length
            for i in range(1, length - 1):
                if labels[i] == 0 and labels[i + 1] == 1:
                    total += 1
                    if compare(model, input_ids, i + 1, device=device) == 'right':
                        correct += 1
                if labels[i] == 1 and labels[i + 1] == 0:
                    total += 1
                    if compare(model, input_ids, i + 1, device=device) == 'left':
                        correct += 1
    evaluation_score = correct / total
    
    with open('experiment_result/temp_score.utf8', 'r') as ff:
        t = ff.readlines()
    with open('experiment_result/SegmentBERT_{}_evaluation_result.txt'.format(dataset), 'a') as rf:
        rf.write(f"----{model_name} Result----\n")
        rf.write(t[-7]+t[-6]+t[-5])
        rf.write("evaluation_score: {} / {} = {}\n\n".format(correct, total, evaluation_score))