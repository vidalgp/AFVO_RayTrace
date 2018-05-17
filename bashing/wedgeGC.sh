#!usr/bin/env bash

cd ../main/
mkdir fig/
mkdir ../../finalwedge

MOD=(b d o a m)
MNAME=("AVO INTENSO" "AVO DEBIL" "OSTRANDER" "MAZZOTTI A" "MAZZOTTI B")
Q=(False True)

k=0

for e in ${Q[@]}
do
    for j in ${MOD[@]}
    do
        echo _______________ MODELO ${MNAME[$k]} Q $e ___________________
        python3 -u wedgemodel_main.py $j $e 100 | tee konsole.log
        cd ../../
        tar -czvf finalwedge/wedge$e$j.tar.gz AFVO_RayTrace/
        cd AFVO_RayTrace/main/
        rm fig/*.png
        ((k++))
    done
    k=0
done
