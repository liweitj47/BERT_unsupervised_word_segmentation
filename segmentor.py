import os
import torch
from transformers import BertTokenizer, BertConfig
from sentence_process import cut_sentence
from text2label import label2text
from SegmentBERT import SegmentBERT

punctuation_ids = {'，': 8024, '。': 511, '（': 8020, '）': 8021, '《': 517, '》': 518, '"': 107, '\'':112, '！': 8013, '、': 510, '℃': 360, '##℃': 8320, '：': 8038, '；': 8039, '？': 8043, '…': 8106, '●': 474, '／': 8027, '①': 405, '②': 406, '③': 407, '④': 408, '⑤': 409, '⑥': 410, '⑦': 411, '⑧': 412, '⑨': 413, '⑩': 414, '＊': 8022, '〈': 515, '〉': 516, '『': 521, '』': 522, '＇': 8019, '｀': 8050, '.': 119, '「': 519, '」': 520}

def seg(text, model):
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    length = len(input_ids)
    input_ids.insert(0, 101) # id of [CLS]
    input_ids.insert(length + 1, 102) # id of [SEP]
    vecs = (model(torch.tensor(input_ids).unsqueeze(0).to(device), mode=1))[0][0]
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

device = torch.device('cuda:2')

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese-pytorch_model/vocab.txt')
bert_config = BertConfig.from_json_file('bert-base-chinese-pytorch_model/bert_config.json')

model = SegmentBERT(bert_config)
model.to(device=device)

dataset = 'pku' # 'pku' or 'msr'
i = 11 # model number
state_dict = torch.load(f'saved_models/SegmentBERT_{dataset}_{i}.pkl', map_location='cpu')
model.load_state_dict(state_dict)
with open(f"experiment_result/SegmentBERT_{dataset}_test_segmented_{i}.utf8", "w") as fo:
    with open(f'dataset/testing/{dataset}_test.utf8', 'r') as f1:
        text = f1.readlines()
        for line in text:
            sentences = cut_sentence(line)
            final_result = str()
            for sentence in sentences:
                if len(sentence) > 510:
                    print(sentence, "length > 510, can't be processed by BERT.")
                    continue
                result = seg(sentence, model)
                final_result += result
            fo.write(final_result)
            fo.write('\n')
os.system(f'perl dataset/scripts/score dataset/gold/{dataset}_training_words.utf8 dataset/gold/{dataset}_test_gold.utf8 experiment_result/SegmentBERT_{dataset}_test_segmented_{i}.utf8 > experiment_result/SegmentBERT_{dataset}_test_score_{i}.utf8')