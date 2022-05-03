#!/usr/bin/env bash
#ISA="XpulpNN"
ISA="XpulpV2"
BW="32bit"
#BW="64bit"
TEST='conv_like'
#TEST='lnq'
#TEST='ap'
#TEST='mp'

precs=(2 4 8)
sgn=(0 1)



cd ${ISA}/${BW}/test
rm -f errors.txt
if [ ${TEST} = 'conv_like' ]; then
# conv, dw, pw, linear_quant
    for si in ${sgn[@]}; do
        for so in ${sgn[@]}; do
            for pi in ${precs[@]}; do
                for po in ${precs[@]}; do
                    for pw in ${precs[@]}; do
                        echo "precision: ${pi}${po}${pw} sign: ${si}${so}"
                        rm -f BUILD/PULP/GCC_RISCV/test/test* && make build -j8 kernel=${pi}${po}${pw} signed=${si}${so}
                        if ! make all run | grep -q "errors: 0"; then
                            echo -e "prec ${pi}${po}${pw}, sgn ${si}${so}\n" >> errors.txt
                        fi
                    done
                done
            done
        done
    done
fi


if [ ${TEST} = 'lnq' ]; then
    # linear_quant
    for si in ${sgn[@]}; do
        for pi in ${precs[@]}; do
            for pw in ${precs[@]}; do
                if ! [ ${pi} = 2 -a ${pw} = 4 ] ; then
                    echo "precision: ${pi}${pw} sign: ${si}${so}"
                    rm -f BUILD/PULP/GCC_RISCV/test/test* && make build -j8 kernel=${pi}${pw} signed=${si}
                    if ! make all run | grep -q "errors: 0"; then
                        echo -e "prec ${pi}${pw}, sgn ${si}\n" >> errors.txt
                    fi
                fi
            done
        done
    done
fi

if [ ${TEST} = 'mp' ]; then
    # maxpool
    for si in ${sgn[@]}; do
            for pi in ${precs[@]}; do
                echo "precision: ${pi} sign: ${si}"
                rm -f BUILD/PULP/GCC_RISCV/test/test* && make build -j8 kernel=${pi} signed=${si}
                if ! make all run | grep -q "errors: 0"; then
                    echo -e "prec ${pi}, sgn ${si}\n" >> errors.txt
                fi
            done
    done
fi


if [ ${TEST} = 'ap' ]; then
    # average pool
    for si in ${sgn[@]}; do
        for so in ${sgn[@]}; do
            for pi in ${precs[@]}; do
                for po in ${precs[@]}; do
                    echo "precision: ${pi}${po} sign: ${si}${so}"
                    rm -f BUILD/PULP/GCC_RISCV/test/test* && make build -j8 kernel=${pi}${po}${pw} signed=${si}${so}
                    if ! make all run | grep -q "errors: 0"; then
                        echo -e "prec ${pi}${po}, sgn ${si}${so}\n" >> errors.txt
                    fi
                done
            done
        done
    done
fi
