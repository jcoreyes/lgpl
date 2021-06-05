# LGPL
Code for running "Guiding Policies with Language via Meta-Learning"

Project Website: https://sites.google.com/view/lgpl

Paper: https://arxiv.org/abs/1811.07882

## Setup
- Clone this repo and add the repo dir to your python path
- conda config --set restore_free_channel true
- conda env create -f docker/environment.yml
- pip install other requirementt manually
- Install https://github.com/justinjfu/doodad
- Download expert policies for minigrid [pickplace](https://drive.google.com/file/d/1jKjdVmtKfv89nlzI47AZrd_a-HButeY0/view?usp=sharing) and [pusher](https://drive.google.com/file/d/1bho0FzFRb_bcHYeN62ePYOhcSCOPOIoh/view?usp=sharing)
- Extract files to <REPO_DIR>/data/exps

## Generating Tasks and Language
- Tasks and their corresponding environment are generated using a context free grammar (using NLTK)
- Existing task data is provided in env_data
- To make modifications to the tasks and generate your own env data, modify either lgpl/language/minigrid_grammar.py or lgpl/language/pusher/pusher_env3.py
- Envs are registered in lgpl/envs/gym_minigrid/envs/six_room_abs_pickplace.py and lgplg/envs/pusher/pusher_env3.py
- For pusher, modify the texture file path in lgpl/envs/assets/pusher_env2.xml to the correct absolute path of lgpl/envs/assets/marble_texture_006.png

## Training Expert Policies
- Saved expert policies will be in data/exps/minigrid/pickplace9 and data/exps/pusher/pusher3v4-expert1 after downloading and extracting
- To train your own expert policies use 
```
python exps/minigrid/train_experts.py
python exps/pusher/train_experts.py
```  

- Minigrid pickplace has 3240 tasks and pusher has 1000 tasks so this can be parallelized on for example 4 CPUs by launching
```
python exps/minigrid/train_experts.py --exp_id_start 0 --exp_id_stride 4
python exps/minigrid/train_experts.py --exp_id_start 1 --exp_id_stride 4
python exps/minigrid/train_experts.py --exp_id_start 2 --exp_id_stride 4
python exps/minigrid/train_experts.py --exp_id_start 3 --exp_id_stride 4
```
such that each python process will train experts on every 4th tasks starting at different offsets.


## Training the Model
- Train the model using:
```
python exps/minigrid/train_lgpl.py 
python exps/pusher/train_lgpl.py
```
which which load in the saved expert policies and check each one is successful before training the model.

- Data is logged by default to data/exps/tmp and can be visualized with https://github.com/vitchyr/viskit
python viskit/frontend.py <exp_dir>
- Number of envs can be reduced with the n_envs arg for quicker debugging.