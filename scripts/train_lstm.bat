@echo off

rem Set your local root directory
set LOC=C:\Users\xk_20\Documents\Code\CS4248

rem Set your data root
set ROOT=%LOC%\StrokeNet
set DATAROOT=%ROOT%\data\NIST\simp

rem Set the bin directory path
set DATABIN=%DATAROOT%\bpe\bin

rem Create necessary directories
mkdir %ROOT%\checkpoints
mkdir %ROOT%\experiments
mkdir %ROOT%\logs

rem Function to configure NIST_zhx_en_2keys
set LANG_LIST=%DATABIN%\langs.file

@REM Simplified Chinese
set LANG_PAIRS="zh-en,zh1-en,zh2-en"
set EVAL_LANG_PAIRS="zh-en,"
set SAMP_MAIN='"main:zh-en":0.0'
set SAMP_TGT='"main:zh1-en":1.0,"main:zh2-en":1.0'
set SAMP_SRC="main:zh1-en":1.0,"main:zh2-en":1.0
set SAMPLE_WEIGHTS="{\"main:zh1-en\": 1.0, \"main:zh2-en\": 1.0, \"main:zh-en\": 0.0}" --virtual-data-size 4991980
set SRC=zh

@REM Traditional Chinese
@REM set LANG_PAIRS="tz-en,tz1-en,tz2-en"
@REM set EVAL_LANG_PAIRS="tz-en,"
@REM set SAMP_MAIN='"main:tz-en":0.0'
@REM set SAMP_TGT='"main:tz1-en":1.0,"main:tz2-en":1.0'
@REM set SAMP_SRC="main:tz1-en":1.0,"main:tz2-en":1.0
@REM set SAMPLE_WEIGHTS="{\"main:tz1-en\": 1.0, \"main:tz2-en\": 1.0, \"main:tz-en\": 0.0}" --virtual-data-size 4991980
@REM set SRC=tz

set EXPTNAME=NIST
set RUN=#0

set CKPT=%ROOT%\checkpoints\%EXPTNAME%
set EXPTDIR=%ROOT%\logs\%EXPTNAME%
mkdir %EXPTDIR%
mkdir %CKPT%

set TASK=translation_multi_simple_epoch_cipher --prime-src %SRC% --prime-tgt en
set LOSS=label_smoothed_cross_entropy_js --js-alpha 5 --js-warmup 500
set ARCH=lstm
set MAX_EPOCH=20
set PATIENCE=5
set MAX_TOK=8000
set UPDATE_FREQ=2

rem Print the experiment details
echo %EXPTNAME%
echo %ARCH%
echo %LOSS%

rem Training begins here
echo Starting training...

@REM fairseq-train --log-format simple --log-interval 1000 ^
@REM echo python %ROOT%\fairseq-cipherdaug\fairseq\train.py

python %ROOT%\fairseq-cipherdaug\train.py --log-format simple --log-interval 1000 ^
    %DATABIN% --save-dir %CKPT% --keep-best-checkpoints 1 ^
    --fp16 --fp16-init-scale 64 ^
    --lang-dict %LANG_LIST% --lang-pairs %LANG_PAIRS% --encoder-langtok tgt ^
    --task %TASK% ^
    --arch %ARCH% --share-all-embeddings ^
    --sampling-weights %SAMPLE_WEIGHTS% ^
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 ^
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 ^
    --dropout 0.3 --weight-decay 0.0001 ^
    --max-epoch %MAX_EPOCH% --patience %PATIENCE% ^
    --keep-last-epochs 5 ^
    --criterion %LOSS% ^
    --label-smoothing 0.1 ^
    --max-tokens %MAX_TOK% --update-freq %UPDATE_FREQ% --eval-bleu ^
    --eval-lang-pairs %EVAL_LANG_PAIRS% ^
    --valid-subset valid --ignore-unused-valid-subsets --batch-size-valid 200 ^
    --eval-bleu-detok moses --eval-bleu-remove-bpe ^
    --eval-bleu-print-samples ^
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric ^
    --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.0, \"max_len_b\": 10}" ^
    >> "%EXPTDIR%\train.%RUN%.log"
