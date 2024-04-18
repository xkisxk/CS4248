@echo off
@REM Edit to path to CS4248Project
set LOC=.\CS4248Project

@REM Simplified Chinese
set DATA=%LOC%\data\NIST\simp
set SCRS="zh zh1 zh2"
set SRC1="zh"
set SRC2="zh1"
set SRC3="zh2"
set TGTS="en zh"
set TGTS1="en"
set TGTS2="zh"

@REM Traditional Chinese
@REM set DATA=%LOC%\StrokeNet\data\NIST\
@REM set SCRS="tz tz1 tz2"
@REM set SRC1="tz"
@REM set SRC2="tz1"
@REM set SRC3="tz2"
@REM set TGTS="en tz"
@REM set TGTS1="en"
@REM set TGTS2="tz"

mkdir "%DATA%\bpe"

@REM :: learning joint bpe in all training data.
type "%DATA%\train.%SRC1%-%TGTS1%.%SRC1%" "%DATA%\train.%SRC2%-%TGTS1%.%SRC2%" "%DATA%\train.%SRC3%-%TGTS1%.%SRC3%" "%DATA%\train.%SRC1%-%TGTS1%.%TGTS1%" > "%DATA%\train.all"

@REM echo Learning joint bpe......
subword-nmt learn-joint-bpe-and-vocab --input "%DATA%\train.all" -s 30000 -o "%DATA%\bpe\joint.code" --min-frequency 50 --write-vocabulary "%DATA%\bpe\joint.vocab"

:: apply bpe
echo Applying bpe......

for %%S in (train validation) do (
    for %%C in (%SRC1% %SRC2% %SRC3%) do (
        for %%T in (%TGTS1% %TGTS2%) do (
            if not %%S == %%T (
                echo Generate %%S.bpe.%%C-%%T.%%C
                subword-nmt apply-bpe -c "%DATA%\bpe\joint.code" < "%DATA%\%%S.%%C-%%T.%%C" > "%DATA%\bpe\%%S.bpe.%%C-%%T.%%C"
                echo Generate %%S.bpe.%%C-%%T.%%T
                subword-nmt apply-bpe -c "%DATA%\bpe\joint.code" < "%DATA%\%%S.%%C-%%T.%%T" > "%DATA%\bpe\%%S.bpe.%%C-%%T.%%T"
            )
        )
    )
)
