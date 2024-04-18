@echo off
set LOC=C:\Users\xk_20\Documents\Code\CS4248

set src=zh
set trg=en
@REM type of model/data that is being worked on
set type=simp
set cipher=_nocipher
set STROKENET=%LOC%\StrokeNet
set ROOT=%LOC%\StrokeNet\data\NIST\%type%\bpe
set TEST=%LOC%\StrokeNet\data\NIST\%type%\test
set CHKPT_LOC=%LOC%\StrokeNet\checkpoints\NIST\lstm_%type%

set T=all

python %STROKENET%\fairseq-cipherdaug\interactive.py %TEST%\%T% --task translation_multi_simple_epoch ^
--path %CHKPT_LOC%\checkpoint_best.pt --beam 5 --remove-bpe ^
--lang-tok-style "multilingual" --source-lang %src% --target-lang %trg% --encoder-langtok "tgt" ^
--lang-dict "%ROOT%\bin\langs.file" ^
--lang-pairs "%src%-%trg%" ^