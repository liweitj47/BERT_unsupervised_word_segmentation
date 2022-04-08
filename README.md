# Unsupervised_CWS_BOPT
This is the source code of ***Unsupervised Chinese Word Segmentation with BERT Oriented Probing and Transformation***.

## To run
First download pre-trained BERT model and put in this directory. Add `"num_labels": 2` in `bert-base-chinese-pytorch_model/bert_config.json`.

Run `train.py` to train models. This may spend lots of time.

Run `evaluation.py` to examine models on development set, which is randomly chosen from training set. Choose the model with highest evaluation_score. (F1-score is just to show that our method is reasonable. It cannot be the standard to choose model)

Run `segmentor.py` to use the model to segment words.

Run `score` script in `dataset/scripts/` to see the recall, precision and F1-score. The usage of it is as follows, which is from *2nd International Chinese Word Segmentation Bakeoff*.

> * Scoring
>
> The script 'score' is used to generate compare two segmentations. The
script takes three arguments:
> 
> 1. The training set word list
> 2. The gold standard segmentation
> 3. The segmented test file
> 
> You must not mix character encodings when invoking the scoring
> script. For example:
> 
> % perl scripts/score gold/pku_training_words.utf8 \
>     gold/pku_test_gold.utf8 test_segmentation.utf8 > score.utf8


