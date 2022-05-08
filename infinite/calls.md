# nre
## prior
### big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=prior data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=prior data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
## joint
### big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=joint data.max_steps_per_epoch=20 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=joint data.max_steps_per_epoch=20 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
# cnre
## prior
### big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=prior data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=prior data.num_simulations=22_528 data.validation_fraction=0.090909 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
## joint
### big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=joint data.max_steps_per_epoch=20 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=joint data.max_steps_per_epoch=20 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun

# old prior calls before fix of validation fraction!!
fix prior by setting
https://www.wolframalpha.com/input?i=solve+system+of+equations&assumption=%7B%22F%22%2C+%22SolveSystemOf2EquationsCalculator%22%2C+%22equation1%22%7D+-%3E%2220+*+1024+%2B+x+*+y+-+y+%3D+0%22&assumption=%22FSelect%22+-%3E+%7B%7B%22SolveSystemOf2EquationsCalculator%22%7D%7D&assumption=%7B%22C%22%2C+%22solve+system+of+equations%22%7D+-%3E+%7B%22Calculator%22%7D&assumption=%7B%22F%22%2C+%22SolveSystemOf2EquationsCalculator%22%2C+%22equation2%22%7D+-%3E%22x+*+y+%3D+2048%22
val_frac = 1/11
samples = 22528
# nre
## prior
### big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=prior data.num_simulations=10_000 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=nre data=prior data.num_simulations=10_000 algorithm.params.state_dict_saving_rate=100 algorithm.params.num_atoms=2,10,25,50,75,100,150,200 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
# cnre
## prior
### big
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=prior data.num_simulations=10_000 algorithm.params.num_blocks=3 algorithm.params.hidden_features=128 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 algorithm.params.gamma=100,10,1,0.1,0.01,0.001 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
### small
python main.py task=two_moons,slcp,gaussian_mixture max_num_epochs=1000 algorithm=cnre data=prior data.num_simulations=10_000 algorithm.params.state_dict_saving_rate=100 algorithm.params.K=1,9,24,49,74,99,149,199 device=cuda:0 hydra/launcher=das5-gpu hydra.launcher.timeout_min=600 --multirun
