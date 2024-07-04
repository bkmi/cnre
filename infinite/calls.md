## joint - Section 3.1
### nre big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=joint data.max_steps_per_epoch=20 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### nre small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=joint data.max_steps_per_epoch=20 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### cnre big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=joint data.max_steps_per_epoch=20 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### cnre small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=joint data.max_steps_per_epoch=20 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun


## prior - Section 3.2
### nre big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=prior data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=snellius-gpu hydra.launcher.timeout_min=600 --multirun
### nre small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=prior data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=snellius-gpu hydra.launcher.timeout_min=600 --multirun
### cnre big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=prior data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=snellius-gpu hydra.launcher.timeout_min=600 --multirun
### cnre small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=prior data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=snellius-gpu hydra.launcher.timeout_min=600 --multirun


## bench - Section 3.3 paragraph 1
### nre big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=bench data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### nre small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=bench data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### cnre big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=bench data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### cnre small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=bench data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun


## actual entire sbibm - Section 3.3 paragraph 2
note: sir and lotka_volterra require julia
note: slcp_distractors may require `data.simulation_batch_size=10000`

python main.py task=lotka_volterra,sir,bernoulli_glm,slcp,gaussian_mixture,gaussian_linear_uniform,two_moons,gaussian_linear,slcp_distractors,bernoulli_glm_raw max_num_epochs=1000 algorithm=cnre data=bench data.num_simulations=1_000,10_000,100_000 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=99 algorithm.params.sample_with=mcmc device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun


## specific curves with fixed validation loss - appendix
### nre
python main.py task=slcp max_num_epochs=1000 algorithm=nre data=bench data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 algorithm.params.val_num_atoms=2 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun

### cnre
python main.py task=slcp max_num_epochs=1000 algorithm=cnre data=bench data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 algorithm.params.val_K=1 algorithm.params.val_gamma=1.0 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
