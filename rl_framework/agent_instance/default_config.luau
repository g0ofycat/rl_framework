--!strict

return {
	batch_size = 64,
	learning_rate = 0.0005,
	gamma = 0.99,
	epsilon = 1.0,
	epsilon_min = 0.05,
	epsilon_decay = 0.99,
	target_update_freq = 10,
	step_limit = 2000,
	step_interval = 0.0,
	interval_per_steps = 10,

	hidden_layers = { 128, 128, 64 },
	dropout_rate = 0.05,
	soft_update_tau = 0.01
}