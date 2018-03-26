# End-to-End Task-Completion Neural Dialogue Systems
*参考 文献
[End-to-End Task-Completion Neural Dialogue Systems](http://arxiv.org/abs/1703.01008)和
[A User Simulator for Task-Completion Dialogues](http://arxiv.org/abs/1612.05688).*
本文档描述了如何运行仿真和不同对话代理（基于规则，命令行，强化学习），更多代理和用户模拟设置方法在文献的Recipe章节。

## Content
* [Data](#数据模块)
* [Parameter](#参数设置模块)
* [Running Dialogue Agents](#运行对话和代理模块)
* [Evaluation](#评估模块)
* [Reference](#参考模块)

## Data
所有的数据都存放在该文件下: ./src/deep_dialog/data

* Movie Knowledge Bases<br/>
`movie_kb.1k.p` ： 94%(for `user_goals_first_turn_template_subsets.v1.p`)<br/>
`movie_kb.v2.p` ： 36%(for `user_goals_first_turn_template_subsets.v1.p`)

* User Goals<br/>
`user_goals_first_turn_template.v2.p` --- user goals extracted from the first user turn<br/>
`user_goals_first_turn_template.part.movie.v1.p` --- a subset of user goals [Please use this one, the upper bound success rate on movie_kb.1k.json is 0.9765.]

* NLG Rule Template<br/>
`dia_act_nl_pairs.v6.json` ：用户模拟器和代理的一些预定义NLG规则模板

* Dialog Act Intent<br/>
`dia_acts.txt`：Intent分类

* Dialog Act Slot<br/>
`slot_set.txt`：Slot分类

## Parameter

### Basic setting

`--agt`:代理Id<br/>
`--usr`: 用户（或模拟器）Id<br/>
`--max_turn`: 对话最大轮数<br/>
`--episodes`: 对话迭代次数<br/>
`--slot_err_prob`: slot错分概率<br/>
`--slot_err_mode`: slot错分为哪个mode<br/>
`--intent_err_prob`: intent错分概率


### Data setting

`--movie_kb_path`:代理方面电影的kb路径<br/>
`--goal_file_path`: 用户目标路径

### Model setting

`--dqn_hidden_size`: DQN代理隐藏层层数t<br/>
`--batch_size`: DQN训练的batch大小<br/>
`--simulation_epoch_size`: 每一次迭代，对话仿真次数<br/>
`--warm_start`: use rule policy to fill the experience replay buffer at the beginning<br/>
`--warm_start_epochs`: 热启动运行对话数量

### Display setting

`--run_mode`: 0 (NL)运行模式; 1(Dia_Act)debug模式; 2(Dia_Act and NL)debug模式; 3(training或者predict)非运行模式<br/>
`--act_level`: 0（Dia_Act级别用户模拟器）; 1（NL级别用户模拟器）<br/>
`--auto_suggest`: 0 （no auto_suggest）; 1（auto_suggest）<br/>
`--cmd_input_mode`: 0（输入方式NL）; 1（输入方式Dia_Act）. (这个参数只针对代理模式为AgentCmd模式时设置)

### Others

`--write_model_dir`:写入模型的目录<br/>
`--trained_model_path`: 训练RL代理模型的目录，也是预测时加载模型的目录.

`--learning_phase`: train/test/all, default is all. You can split the user goal set into train and test set, or do not split (all); We introduce some randomness at the first sampled user action, even for the same user goal, the generated dialogue might be different.<br/>

## Running Dialogue Agents

### Rule Agent
```sh
python run.py --agt 5 --usr 1 --max_turn 40
	      --episodes 150
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --intent_err_prob 0.00
	      --slot_err_prob 0.00
	      --episodes 500
	      --act_level 0
```

### Cmd Agent
NL Input
```sh
python run.py --agt 0 --usr 1 --max_turn 40
	      --episodes 150
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --intent_err_prob 0.00
	      --slot_err_prob 0.00
	      --episodes 500
	      --act_level 0
	      --run_mode 0
	      --cmd_input_mode 0
```
Dia_Act Input
```sh
python run.py --agt 0 --usr 1 --max_turn 40
	      --episodes 150
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p 
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --intent_err_prob 0.00
	      --slot_err_prob 0.00
	      --episodes 500
	      --act_level 0
	      --run_mode 0
	      --cmd_input_mode 1
```

### End2End RL Agent
Train End2End RL Agent without NLU and NLG (with simulated noise in NLU)
```sh
python run.py --agt 9 --usr 1 --max_turn 40
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --dqn_hidden_size 80
	      --experience_replay_pool_size 1000
	      --episodes 500
	      --simulation_epoch_size 100
	      --write_model_dir ./deep_dialog/checkpoints/rl_agent/
	      --run_mode 3
	      --act_level 0
	      --slot_err_prob 0.00
	      --intent_err_prob 0.00
	      --batch_size 16
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --warm_start 1
	      --warm_start_epochs 120
```
Train End2End RL Agent with NLU and NLG
```sh
python run.py --agt 9 --usr 1 --max_turn 40
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --dqn_hidden_size 80
	      --experience_replay_pool_size 1000
	      --episodes 500
	      --simulation_epoch_size 100
	      --write_model_dir ./deep_dialog/checkpoints/rl_agent/
	      --run_mode 3
	      --act_level 1
	      --slot_err_prob 0.00
	      --intent_err_prob 0.00
	      --batch_size 16
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --warm_start 1
	      --warm_start_epochs 120
```
Test RL Agent with N dialogues:
```sh
python run.py --agt 9 --usr 1 --max_turn 40
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --dqn_hidden_size 80
	      --experience_replay_pool_size 1000
	      --episodes 300 
	      --simulation_epoch_size 100
	      --write_model_dir ./deep_dialog/checkpoints/rl_agent/
	      --slot_err_prob 0.00
	      --intent_err_prob 0.00
	      --batch_size 16
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --trained_model_path ./deep_dialog/checkpoints/rl_agent/noe2e/agt_9_478_500_0.98000.p
	      --run_mode 3
```

## Evaluation
To evaluate the performance of agents, three metrics are available: success rate, average reward, average turns. Here we show the learning curve with success rate.

1. Plotting Learning Curve
``` python draw_learning_curve.py --result_file ./deep_dialog/checkpoints/rl_agent/noe2e/agt_9_performance_records.json```
2. Pull out the numbers and draw the curves in Excel

## Reference

Main papers to be cited
```
@inproceedings{li2017end,
  title={End-to-End Task-Completion Neural Dialogue Systems},
  author={Li, Xuijun and Chen, Yun-Nung and Li, Lihong and Gao, Jianfeng and Celikyilmaz, Asli},
  booktitle={Proceedings of The 8th International Joint Conference on Natural Language Processing},
  year={2017}
}

@article{li2016user,
  title={A User Simulator for Task-Completion Dialogues},
  author={Li, Xiujun and Lipton, Zachary C and Dhingra, Bhuwan and Li, Lihong and Gao, Jianfeng and Chen, Yun-Nung},
  journal={arXiv preprint arXiv:1612.05688},
  year={2016}
}
