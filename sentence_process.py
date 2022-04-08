import re

def cut_sentence(text):
    '''
    Cut paragraph into sentences because BERT has a maximum input length of 512.
    '''
    sentences = re.split(r"(？|。|！|……|“|”|‘|’)", text)
    sentences.append("")
    new_sentences = []
    for i in zip(sentences[0::2], sentences[1::2]):
        if ('“' in i) or ('”' in i) or ('‘' in i) or ('’' in i):
            new_sentences.append(i[0])
            new_sentences.append(i[1])
        else:
             new_sentences.append("".join(i))
    new_sentences = [i.strip() for i in new_sentences if len(i.strip()) > 0]

    return new_sentences

def unk_process(text, result_text):
    '''
    Process '[UNK]' because there are some tokens unknown to BERT. 

    Process Uppercase letters, because BERT encode both uppercase letters
    and lowercase letters into the same tokens, which will be decoded into 
    only lowercase letters.

    *** This is not post-process ***
    '''
    # process [UNK]
    if '[UNK]' in result_text:
        origin_tokens = []
        for token in text:
            if (token not in result_text) and not re.match('[a-z]|[A-Z]|[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]', token):
                origin_tokens.append(token)
        for token in origin_tokens:
            result_text = result_text.replace('[UNK]', token, 1)

    # process uppercase and lowercase letters
    letters = []
    for i in text:
        if (re.match('[a-z]|[A-Z]|[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]', i)):
            letters.append(i)
    p = 0
    s1 = list(result_text)
    for i in range(len(result_text)):
        if (re.match('[a-z]|[A-Z]|[ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ]', s1[i])):
            s1[i] = letters[p]
            p += 1
    result_text = ''.join(s1)
    return result_text

def process_number_english(text):
    '''
    Process numbers and English words, since BERT may separate them. 

    For example, BERT may separate '1812' into '181' and '2', which should be processd by following code.
    '''
    while True:
        group = re.search('([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])  ##([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])', text)
        if group is None:
            break
        text = text.replace(group.group(), group.group().replace('  ##', ''), 1)
    
    while True:
        group = re.search('([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])  ((\.|．)  )+([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])', text)
        if group is None:
            break
        text = text.replace(group.group(), group.group().replace('  ', ''), 1)

    while True:
        group = re.search('([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])  (\.|．)([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])', text)
        if group is None:
            break
        text = text.replace(group.group(), group.group().replace('  .', '.').replace('  ．', '．'), 1)

    while True:
        group = re.search('([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])(\.|．)  ([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])', text)
        if group is None:
            break
        text = text.replace(group.group(), group.group().replace('.  ', '.').replace('．  ', '．'), 1)

    while True:
        group = re.search('([0-9]|[０１２３４５６７８９])  ％', text)
        if group is None:
            break
        text = text.replace(group.group(), group.group().replace('  ', ''), 1)

    while True:
        group = re.search('  ##([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])', text)
        if group is None:
            break
        text = text.replace(group.group(), group.group().replace('##', ''), 1)

    while True:
        group = re.search('([0-9]|[a-z]|[A-Z]|[０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ])  ##', text)
        if group is None:
            break
        text = text.replace(group.group(), group.group().replace('##', ''), 1)

    while True:
        group = re.search('([0-9]|[０１２３４５６７８９])  (年|月|日|时)', text)
        if group is None:
            break
        text = text.replace(group.group(), group.group().replace('  ', ''), 1)
    
    return text