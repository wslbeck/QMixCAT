#!/bin/bash

python ./code/evaluation.py --mix_scp /googol/shnulab/wangxuefei/data_wsl/VCTK_16k/path/valid_dir/noisy.scp --ref_scp /googol/shnulab/wangxuefei/data_wsl/VCTK_16k/path/valid_dir/clean.scp --log_file enhance.log -m ./base_dir
