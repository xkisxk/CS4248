@echo off
set LOC=.\CS4248Project
set ROOT=%LOC%


@REM Simplified Chinese
set DATAROOT=%ROOT%\data\NIST\simp\bpe
set SRC1=zh
set SRC2=zh1
set SRC3=zh2
set TGTS1=en
set TGTS2=zh

@REM Traditional Chinese
@REM set DATAROOT=%ROOT%\data\NIST\trad_500\bpe
@REM set SRC1=tz
@REM set SRC2=tz1
@REM set SRC3=tz2
@REM set TGTS1=en
@REM set TGTS2=tz

set DATABIN=%DATAROOT%\bin
mkdir %DATABIN%
set DICT=jointdict.txt

echo Generating joined dictionary for all languages based on BPE..
:: strip the first three special tokens and append fake counts for each vocabulary
(for /f "tokens=1" %%a in ('type "%DATAROOT%\joint.vocab"') do echo %%a 100) > "%DATABIN%\%DICT%"

echo binarizing pairwise langs ..
for %%S in (%SRC1% %SRC2% %SRC3%) do (
    for %%T in (%TGTS1% %TGTS2%) do (
        if not %%S == %%T (
            echo binarizing data %%S-%%T data..
            @REM fairseq-preprocess --source-lang %%S --target-lang %%T ^
            python %ROOT%\fairseq-cipherdaug\preprocess.py --source-lang %%S --target-lang %%T ^
                --destdir %DATABIN% ^
                --trainpref %DATAROOT%\train.bpe.%%S-%%T ^
                --validpref %DATAROOT%\validation.bpe.%%S-%%T ^
                --srcdict %DATABIN%\%DICT% --tgtdict %DATABIN%\%DICT% ^
                --workers 10
        )
    )
)

rem Define the contents of the langs file
echo "Creating langs file based on binarised dicts .."
echo %SRC1% >> %DATABIN%\langs.file
echo %SRC2% >> %DATABIN%\langs.file
echo %SRC3% >> %DATABIN%\langs.file
echo %TGTS1% >> %DATABIN%\langs.file
echo "--> %DATABIN%\langs.file"