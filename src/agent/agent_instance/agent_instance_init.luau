--!strict
--!native
--!optimize 2

local agent_instance = {}

agent_instance.__index = agent_instance

--=========================
-- // MODULES
--=========================

agent_instance.Modules = {
	ts_database = require("../../db/ts_database/ts_database_init"),
	rl_env = require("../../env/rl_env/rl_env_init"),
	lru_cache = require("../../env/utility/lru_cache/lru_cache_init"),
	MLP = require("../../nn/MLP/MLP_Init"),

	default_config = require("./default_config"),
	config = require("../../config")
}

--=========================
-- // TYPES
--=========================

local Types = require("./Types")

export type agent_instance_constructor_type = typeof(setmetatable({} :: Types.agent_type, agent_instance))

--========================
-- // DB INIT
--========================

agent_instance.Modules.ts_database.set_api_key(agent_instance.Modules.config.strings.db_api_key)

agent_instance.Modules.ts_database.load_version("Agent_Data")

--=========================
-- // CONSTRUCTOR
--=========================

-- new(): Create a new agent
-- @param state_size: The size of the state
-- @param action_size: The size of the action
-- @param action_map: Dictionary map for action to reward
-- @param lru_cache_capacity: Cache capacity for LRU
-- @param callbacks?: Callbacks for environment events
-- @param env_variables?: Environment variables
-- @param config?: Types.ConfigType?
-- @return agent_instance_constructor_type
function agent_instance.new(state_size: number, action_size: number, action_map: { [any]: number }, lru_cache_capacity: number, callbacks: Types.callback_table?, env_variables: any?, config: Types.ConfigType?): agent_instance_constructor_type
	local new_config = table.clone(agent_instance.Modules.default_config)

	if config then
		for k, v in config do
			new_config[k] = v
		end
	end

	local self = {}

	self.policy_net = agent_instance.Modules.MLP.new(state_size, new_config.hidden_layers, action_size, new_config.dropout_rate)
	self.target_net = agent_instance.Modules.MLP.new(state_size, new_config.hidden_layers, action_size, new_config.dropout_rate)

	self.env = agent_instance.Modules.rl_env.new(action_map, lru_cache_capacity, callbacks, env_variables)

	self.history = {}
	self.history.memory = agent_instance.Modules.lru_cache.new(lru_cache_capacity)

	self.config = new_config

	self.action_size = action_size

	return setmetatable(self, agent_instance)
end

--=========================
-- // ACTION SELECTION
--=========================

-- select_action(): Selects an action based on the current state
-- @param state: The current state
-- @param training: Whether or not to enable random training
-- @return number: The selected action
function agent_instance:select_action(state: { number }, training: boolean): number
	if training and math.random() < self.config.epsilon then
		return math.random(1, self.action_size)
	else
		return self:get_best_action(state)
	end
end

-- get_best_action(): Get the best action (Forward Pass)
-- @param state: The current state
-- @return number: Best Action
function agent_instance:get_best_action(state: { number }): number
	local _, _, q_values = self.policy_net:ForwardPropagation(state, false)

	local best_action = 1
	local max_q = q_values[1]

	for i = 2, #q_values do
		if q_values[i] :: number > max_q then
			max_q = q_values[i]
			best_action = i
		end
	end

	return best_action
end

-- create_inference_thread(): Creates a new inference thread
-- @param get_state: Function to return the current state
-- @param on_action: Callback function to call when an action is selected (action_index)
-- @return thread: The thread loop
function agent_instance:create_inference_thread(get_state: () -> { number }, on_action: (action_index: number) -> ()): thread
	return task.spawn(function()
		local index = 0

		while not self.env.done do
			local state = get_state()

			local action = self:get_best_action(state)

			on_action(action)

			index += 1

			if index % agent_instance.Modules.default_config.interval_per_steps == 0 then
				task.wait(self.config.step_interval)
			end
		end
	end)
end

-- index_to_action(): Converts an index to an action
-- @param action_index: The index of the action
-- @return string: The action
function agent_instance:index_to_action(action_index: number): string
	return self.env.action_list[action_index]
end

--=========================
-- // TRAINING
--=========================

-- add_to_memory(): Adds a new memory to the memory list
-- @param state: The current state
-- @param action: The action taken
-- @param reward: The reward received
-- @param next_state: The next state
-- @param done: If the enviroment is done
function agent_instance:add_to_memory(state: { number }, action: number, reward: number, next_state: { number }, done: boolean): ()
	self.history.memory:put_array({ state = state, action = action, reward = reward, next_state = next_state, done = done })
end

-- train_step(): Trains the agent for a single step
function agent_instance:train_step(): ()
	local memory = self.history.memory

	if memory:size() < self.config.batch_size :: number then return end

	local indices = {}

	for i = 1, #memory do
		indices[i] = i
	end

	for i = #indices, 2, -1 do
		local j = math.random(1, i)
		indices[i], indices[j] = indices[j], indices[i]
	end

	local batch = {}

	for i = 1, self.config.batch_size do
		table.insert(batch, memory[indices[i]])
	end

	for _, dict in batch do
		local hidden_inputs, hidden_outputs, current_q_values = self.policy_net:ForwardPropagation(dict.state, true)
		local target_q_values = table.clone(current_q_values)

		local target = dict.reward

		if not dict.done then
			local _, _, next_q = self.target_net:ForwardPropagation(dict.next_state, false)

			local max_next_q = math.max(table.unpack(next_q))

			target = dict.reward :: number + self.config.gamma :: number * max_next_q
		end

		target_q_values[dict.action] = target

		self.policy_net:BackPropagation(
			dict.state,
			target_q_values,
			hidden_inputs,
			hidden_outputs,
			current_q_values,
			self.config.learning_rate
		)
	end
end

-- decay_epsilon(): Decays the epsilon value
function agent_instance:decay_epsilon(): ()
	local epsilon = self.config.epsilon

	if epsilon > self.config.epsilon_min :: number then
		self.config.epsilon = math.max(self.config.epsilon_min, self.config.epsilon * self.config.epsilon_decay)
	end
end

-- update_target_network(): Update the target network weights with the policy network weights
function agent_instance:update_target_network(): ()
	self.policy_net:SoftUpdate(self.target_net, self.config.soft_update_tau)
end

-- get_env_variables(): Return the env variable table
-- @return any?
function agent_instance:get_env_variables(): any?
	return self.env.env_variables
end

-- return_config(): Return the config table
function agent_instance:return_config(): any
	return self.config
end

--=========================
-- // SAVING / LOADING
--=========================

-- save_to_db(): Saves the networks to the database
-- @param name?: Custom name, if any
-- @return id: The ID of the inserted networks
function agent_instance:save_to_db(name: any?): number
	local id = agent_instance.Modules.ts_database.insert({ name = name, data = { policy_net = self.policy_net:ExportData(), target_net = self.target_net:ExportData() } })

	return id
end

-- load_from_db(): Loads the networks from the database
-- @param id: The ID of the data to load
function agent_instance:load_from_db(id: number): ()
	local json_body = agent_instance.Modules.ts_database.get(id)

	if not json_body then
		warn("load_from_db(): Failed to load from database")
		return 
	end

	self.policy_net:ImportData(json_body.data.policy_net)
	self.target_net:ImportData(json_body.data.target_net)
end

--=========================
-- // EPISODES
--=========================

-- run_episode(): Runs a single episode
-- @param step_callback?: The function to call each step
-- @return total_reward: The total reward for the episode
function agent_instance:run_episode(step_callback: ((observation: { number }) -> ())?): number
	local observation = self.env:reset()
	local total_reward = 0
	local current_step = 0
	local index = 0

	while not self.env.done and current_step < self.config.step_limit do
		if step_callback then
			step_callback(observation)
		end

		local action_index = self:select_action(observation, true)

		local result = self.env:step(action_index)

		local next_observation = result.observation
		local reward = result.score

		self:add_to_memory(observation, action_index, reward, next_observation, result.done)
		self:train_step()

		observation = next_observation
		total_reward += reward

		current_step += 1

		index += 1

		if index % agent_instance.Modules.default_config.interval_per_steps == 0 then
			task.wait(self.config.step_interval)
		end
	end

	self:decay_epsilon()

	return total_reward
end

-- run_episodes(): Runs multiple episodes
-- @param num_episodes: The number of episodes to run
-- @param episode_callback?: The function to call after each episode (Current Episode, Reward)
-- @param step_callback?: The function to call each step (observation)
function agent_instance:run_episodes(
	num_episodes: number,
	episode_callback: ((episode: number, reward: number) -> ())?,
	step_callback: ((observation: { number }) -> ())?
): ()
	for episode = 1, num_episodes do
		local total_reward = self:run_episode(step_callback)

		if episode_callback then
			episode_callback(episode, total_reward)
		end

		if episode % self.config.target_update_freq == 0 then
			self:update_target_network()
		end
	end
end

return agent_instance