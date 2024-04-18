@echo off
set LOC=C:\Users\xk_20\Documents\Code\CS4248

set src=zh
set trg=en
@REM type of model/data that is being worked on
set type=simp
set cipher=
set STROKENET=%LOC%\StrokeNet
set ROOT=%LOC%\StrokeNet\data\NIST\%type%\bpe
set TEST=%LOC%\StrokeNet\data\NIST\%type%\test
set CHKPT_LOC=%LOC%\StrokeNet\checkpoints\NIST\simp

set T=all
set RESULT=%STROKENET%\results\%type%%cipher%\%T%
mkdir %RESULT%
for %%c in (_best) do (
    @REM apply bpe
    echo Applying BPE
    subword-nmt apply-bpe -c %ROOT%\joint.code < %TEST%\%T%\test.%src%-%trg%.%src% > %TEST%\%T%\test.bpe.%src%
    subword-nmt apply-bpe -c %ROOT%\joint.code < %TEST%\%T%\test.%src%-%trg%.%trg% > %TEST%\%T%\test.bpe.%trg%

    @REM binarize data
    echo Binarizing Data
    python %STROKENET%\fairseq-cipherdaug\preprocess.py --testpref %TEST%\%T%\test.bpe -s %src% -t %trg% ^
    --srcdict %ROOT%\bin\jointdict.txt --tgtdict %ROOT%\bin\jointdict.txt ^
    --destdir %TEST%\%T%

    if exist %RESULT%\generate-test.txt (
        del %RESULT%\generate-test.txt
    )
    echo Start evaluating checkpoint%%c
    python %STROKENET%\fairseq-cipherdaug\generate.py %TEST%\%T% --task translation_multi_simple_epoch ^
    --lang-tok-style "multilingual" --source-lang %src% --target-lang %trg% --encoder-langtok "tgt" ^
    --lang-dict "%ROOT%\bin\langs.file" ^
    --lang-pairs "%src%-%trg%" ^
    --path %CHKPT_LOC%\checkpoint%%c.pt ^
    --batch-size 16 --beam 5 --remove-bpe ^
    --results-path %RESULT%
    
    @REM Concatenate files and filter lines starting with "H"
    echo Extracting
    python %STROKENET%\results\extract.py --path %RESULT%

    @REM run sacrebleu
    echo Running Sacrebleu
    sacrebleu -tok none -s none ^
    %TEST%\%T%\test.%src%-%trg%.%trg% ^
    < %RESULT%\generate-test.hyp > %RESULT%\generate-test.score
)
    @REM %TEST%\%T%\test.%src%-%trg%.%trg%1 ^
    @REM %TEST%\%T%\test.%src%-%trg%.%trg%2 ^
    @REM %TEST%\%T%\test.%src%-%trg%.%trg%3 ^
