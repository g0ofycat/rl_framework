--!strict

return {
	batch_size = 128,
	learning_rate = 0.0005,
	gamma = 0.99,
	epsilon = 1.0,
	epsilon_min = 0.05,
	epsilon_decay = 0.995,
	target_update_freq = 10,
	step_limit = 50,
	step_interval = 0.5,
	interval_per_steps = 1,

	hidden_layers = { 64, 64 },
	dropout_rate = 0.05,
	soft_update_tau = 0.01
}