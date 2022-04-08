from transformers import BertTokenizer
from sentence_process import unk_process, process_number_english

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese-pytorch_model/vocab.txt')

def label2text_ids(text_ids, labels):
    '''
    Input: a list of token ids and a list of labels,
    Return: a list of words.

    e.g. text_ids=[x1, x2, x3, x4, x5, x6], labels=[0, 0, 1, 0, 0, 1]
         --> [[x1], [x2, x3], [x4], [x5, x6]]
    '''
    assert len(text_ids) == len(labels)
    word_ids = []
    p = 0
    for i in range(len(labels)):
        if labels[i] == 0: # label 'B'
            word_ids.append(text_ids[p:i])
            p = i
        if i == len(labels) - 1:
            word_ids.append(text_ids[p:])
    return word_ids[1:]

def text_ids2label(word_ids_list):
    '''
    Input: a list of words,
    Return: a list of labels.

    e.g. word_ids_list=[[x1], [x2, x3], [x4], [x5, x6]]
         --> [0, 0, 1, 0, 0, 1]
    '''
    labels = []
    for word in word_ids_list:
        if len(word) == 0:
            continue
        labels.append(0)
        if len(word) == 1:
            continue
        for i in range(len(word) - 1):
            labels.append(1)
    return labels

def label2text(text, labels):
    '''
    Input: a string and a list of labels,
    Return: segmented text.

    e.g. text='我喜欢写代码' (I love writing code) , labels=[0, 0, 1, 0, 0, 1]
         --> '我  喜欢  写  代码'
    '''
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    word_ids = label2text_ids(token_ids, labels)
    words = [tokenizer.decode(ids) for ids in word_ids]
    for i in range(len(words)):
        words[i] = words[i].replace(' ','')
    res = str()
    for word in words:
        res += (word + '  ')
    res = process_number_english(res)
    res = res.replace('  ##', '')
    res = unk_process(text, res)
    for precess_time in range(2):
        res = res.replace('—  —', '——') # pku dataset: process dash '————'
        res = res.replace('…  …', '……') # pku dataset: process suspension points '…………'
        # res = res.replace('……', '…  …') # msr dataset: process suspension points '…………'
    return res

def text2label(seg_text):
    '''
    Input: segmented text
    Return: a list of labels

    e.g. seg_text='我  喜欢  写  代码' (I love writing code)
         --> [0, 0, 1, 0, 0, 1]
    '''
    words = seg_text.split('  ')
    word_ids = [tokenizer.encode(word, add_special_tokens=False) for word in words]
    return text_ids2label(word_ids)
