--!strict

export type rl_type<lru_type> = {
	score: number,
	done: boolean,
	action_map: { [any]: number },
	current_action: string,
	info: { [number]: any },
	env_variables: any?,
	lru_actions: lru_type,

	on_reset: ((self: any) -> ())?,
	on_step: ((self: any, action_index: number) -> (number, boolean))?,
	on_action: ((self: any, action_index: number) -> ())?,
	observation_overwrite: ((self: any) -> observation_type)?,

	action_list: { string }
}

export type step_return = {
	done: boolean,
	info: { [number]: any },
	observation: observation_type
}

export type rl_env_exported_data = {
	score: number,
	done: boolean,
	action_map: { [any]: number },
	current_action: string,
	info: { [number]: any },
	exported_lru_data: any
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