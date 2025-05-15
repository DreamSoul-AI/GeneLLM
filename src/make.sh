#!/bin/bash

# base all
python make.py --task_name all --subset_name all --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name all --subset_name all --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name EMP --subset_name all --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name EMP --subset_name all --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name mouse --subset_name all --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name mouse --subset_name all --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name promcore --subset_name all --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name promcore --subset_name all --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name prom300 --subset_name all --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name prom300 --subset_name all --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name splice --subset_name all --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name splice --subset_name all --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name tf --subset_name all --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name tf --subset_name all --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name virus --subset_name all --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name virus --subset_name all --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

# base split
python make.py --task_name EMP --subset_name split --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name EMP --subset_name split --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name mouse --subset_name split --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name mouse --subset_name split --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name promcore --subset_name split --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name promcore --subset_name split --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name prom300 --subset_name split --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name prom300 --subset_name split --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name splice --subset_name split --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name splice --subset_name split --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name tf --subset_name split --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name tf --subset_name split --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name virus --subset_name split --mode base --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name virus --subset_name split --mode base --run test --num_experiments 1 --round 1 --num_gpus 1

# embedding all
python make.py --task_name all --subset_name all --mode embedding --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name all --subset_name all --mode embedding --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name EMP --subset_name all --mode embedding --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name EMP --subset_name all --mode embedding --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name mouse --subset_name all --mode embedding --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name mouse --subset_name all --mode embedding --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name promcore --subset_name all --mode embedding --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name promcore --subset_name all --mode embedding --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name prom300 --subset_name all --mode embedding --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name prom300 --subset_name all --mode embedding --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name splice --subset_name all --mode embedding --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name splice --subset_name all --mode embedding --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name tf --subset_name all --mode embedding --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name tf --subset_name all --mode embedding --run test --num_experiments 1 --round 1 --num_gpus 1

python make.py --task_name virus --subset_name all --mode embedding --run train --num_experiments 1 --round 1 --num_gpus 1
python make.py --task_name virus --subset_name all --mode embedding --run test --num_experiments 1 --round 1 --num_gpus 1