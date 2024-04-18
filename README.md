The code here is adapted from [StrokeNet](https://github.com/zjwang21/StrokeNet?tab=readme-ov-file).

## Requirements
1. Install PyTorch with CUDA.
2. Install the requirements,with ```pip install -r requirements.txt```
3. Change directory to fairseq-cipherdaug, ```cd fairseq-cipherdaug```
4. Run ```pip install --editable ./```

## Preprocessing
### Training Data
To generate the Training Data, run the ```iwslt_train.ipynb``` notebook, it will generate the Latinized strokes and the ciphertext as well.

### Testing Data
To generate the Testing Data, run the ```iwslt_test.ipynb``` notebook, it will generate the Latinized strokes and the ciphertext as well.


### Training BPE, applying BPE and binarizing data
Because we train on the BPE text, we need to train the BPE using subword-nmt. For learning and applying BPE algorithm on all relevant files at once, use the `bpe.bat` located in ./scripts. To change from simplified to traditional we need to comment/uncomment the lines that are labelled as traditional.
Number of BPE merge operations can be changed in bash file. This part could last for minutes, wait patiently for it to finish. 
Then use `multi_binarize.bat` to generate joint multilingual dictionary and binary files for fairseq to use. With this, we are ready to start training our model.

## Training
Because we trained the models using Windows, all the scripts are in batch. Use `train_lstm.bat` to train the LSTM model.
Part of the key parameters:
```
fairseq-train $DATABIN --save-dir ${CKPT} \
    --lang-dict "${LANG_LIST}" --lang-pairs "${LANG_PAIRS}" \
    --eval-lang-pairs ${EVAL_LANG_PAIRS} \
    --task ${TASK} \                                         
    --arch transformer --share-all-embeddings \                 # Weight tying
    --criterion ${LOSS} --label-smoothing 0.1 \            
    --valid-subset valid --ignore-unused-valid-subsets --batch-size-valid 200 \
```

To switch between simplified and traditional, comment/uncomment the repsective lines.

### Evaluation
The evaluation script is ```eval.bat``` to generate the predictions, then we can run ```sacrebleu.ipynb``` in ./scripts to get the SacreBLEU scores that are available in our report.