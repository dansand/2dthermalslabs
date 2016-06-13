#!/bin/bash
for i in {1..6}; do 
    gtimeout 320s  mpirun -np 8 python base_model.py
done
