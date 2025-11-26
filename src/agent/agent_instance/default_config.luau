--!strict

return {
	batch_size = 4,
	learning_rate = 0.001,
	gamma = 0.99,
	epsilon = 1.0,
	epsilon_min = 0.01,
	epsilon_decay = 0.999,
	target_update_freq = 10,
	step_limit = 10,
	step_interval = 0.0,
	interval_per_steps = 1,

	hidden_neurons = 16,
	dropout_rate = 0.01,
	soft_update_tau = 0.01
}