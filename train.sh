#!/bin/csh

python3 Snake.py -P "DQN();Avoid(epsilon=0.3);Avoid(epsilon=0.2);Avoid(epsilon=0.1);Avoid(epsilon=0)" -D 50000 -s 5000 -l log/log -r 0 -plt 0.05 -pat 0.01 -pit 10
