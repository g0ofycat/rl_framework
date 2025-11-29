--!strict

export type ConfigType = {
	batch_size: number,
	learning_rate: number,
	gamma: number,
	epsilon: number,
	epsilon_min: number,
	epsilon_decay: number,
	target_update_freq: number,
	step_limit: number,
	step_interval: number,
	interval_per_steps: number,

	hidden_layers: { number },
	dropout_rate: number,
	soft_update_tau: number
}

export type agent_type = {
	policy_net: any,
	target_net: any,
	env: any,

	history: {
		memory: any
	},

	config: any,

	action_size: number
}

export type observation_type = { any } | {
	score: number,
	current_action: string,
	lru_data: any
}

export type callback_table = {
	on_reset: ((self: any) -> ())?,
	on_step: ((self: any, action_index: number) -> (number, boolean))?,
	on_action: ((self: any, action_index: number) -> ())?,
	observation_overwrite: ((self: any) -> observation_type)?
}

return nil