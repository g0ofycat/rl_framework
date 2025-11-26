--!strict

return {
	batch_size = 1,
	learning_rate = 0.005,
	gamma = 0.99,
	epsilon = 1.0,
	epsilon_min = 0.05,
	epsilon_decay = 0.995,
	target_update_freq = 10,
	step_limit = 10,
	step_interval = 0.001,

	hidden_neurons = 128,
	dropout_rate = 0.01,
	soft_update_tau = 0.01
}