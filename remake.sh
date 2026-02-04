#!/bin/bash

xmake clean -a

xmake

xmake install 

pip install ./python/

python test/test_infer.py --model /home/blackghost/models/DeepSeek-R1-Distill-Qwen-1.5B --test