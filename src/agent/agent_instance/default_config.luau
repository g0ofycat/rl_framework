--!strict

return {
	batch_size = 32,
	learning_rate = 0.001,
	gamma = 0.95,
	epsilon = 1.0,
	epsilon_min = 0.05,
	epsilon_decay = 0.9995,
	target_update_freq = 20,
	step_limit = 200,
	step_interval = 0.0,
	interval_per_steps = 100,

	hidden_layers = { 64, 64, 64 },
	dropout_rate = 0.1,
	soft_update_tau = 0.05
}