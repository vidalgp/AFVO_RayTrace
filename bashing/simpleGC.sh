#!usr/bin/env bash

cd ../main/
mkdir fig/
mkdir ../../finalsimple

MOD=(b d o a m)
MNAME=("AVO INTENSO" "AVO DEBIL" "OSTRANDER" "MAZZOTTI A" "MAZZOTTI B")
Q=(False True)

k=0

for e in ${Q[@]}
do
    for j in ${MOD[@]}
    do
        echo _______________ MODELO ${MNAME[$k]} Q $e ___________________
        python3 -u simplemodel_main.py $j $e | tee konsole.log
        cd ../../
        tar -czvf finalsimple/simple_$e$j.tar.gz USB-AFVA/
        cd USB-AFVA/main/
        rm fig/*.png
        ((k++))
    done
    k=0
done
