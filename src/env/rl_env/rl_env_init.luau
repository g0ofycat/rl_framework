--!strict

local rl_env = {}

rl_env.__index = rl_env

--=========================
-- // MODULES
--=========================

local lru_cache = require("../utility/lru_cache/lru_cache_init")

--=========================
-- // TYPES
--=========================

local Types = require("./Types")

export type rl_env_constructor_type<lru_cache> = typeof(setmetatable({} :: Types.rl_type<lru_cache>, rl_env))

--=========================
-- // CONSTRUCTOR
--=========================

-- new(): New Reinforcement Learning Environment
-- @param action_map: Dictionary map for action to score
-- @param lru_cache_capacity: Cache capacity for LRU
-- @param callbacks?: Callbacks for environment events
-- @param env_variables?: Environment variables
-- @return rl_env_constructor_type<lru_cache.lru_cache_constructor_type<string, number>>
function rl_env.new(action_map: { [any]: number }, lru_cache_capacity: number, callbacks: Types.callback_table?, env_variables: any?): rl_env_constructor_type<lru_cache.lru_cache_constructor_type<string, number>>
	local self = {}

	self.score = 0
	self.done = false
	self.action_map = action_map
	self.current_action = ""
	self.info = {}
	self.env_variables = env_variables
	self.lru_actions = lru_cache.new(lru_cache_capacity)

	self.on_reset = callbacks and callbacks.on_reset or nil
	self.on_step = callbacks and callbacks.on_step or nil
	self.on_action = callbacks and callbacks.on_action or nil
	self.observation_overwrite = callbacks and callbacks.observation_overwrite or nil

	self.action_list = {}

	for action_name, _ in action_map do
		table.insert(self.action_list, action_name)
	end

	return setmetatable(self, rl_env)
end

--=========================
-- // MAIN API
--=========================

-- add_action(): Adds an action to the cache
-- @param action: The action to add
-- @param action_index?: The index of the action
-- @param info?: Any extra info / data
function rl_env:add_action(action: string, action_index: number?, info: any?): ()
	local score = self.score :: number
	local score_action = self.action_map[action] or 0 :: number

	self.info[score] = info

	self.current_action = action

	score += score_action

	self.score = score

	if self.on_action then
		self:on_action(action_index)
	end

	self.lru_actions:put(action, score)
end

-- reset(): Reset the enviroment
-- @return Types.observation_type
function rl_env:reset(): Types.observation_type
	self.score = 0
	self.done = false
	self.current_action = ""
	self.info = {}
	self.lru_actions:reset()

	if self.on_reset then
		self:on_reset()
	end

	return self:get_observation()
end

--=========================
-- // ENV API
--=========================

-- get_observation(): Get the current observation / state
-- @return Types.observation_type
function rl_env:get_observation(): Types.observation_type
	if self.observation_overwrite then
		return self:observation_overwrite()
	else
		return {
			score = self.score,
			current_action = self.current_action,
			lru_data = self.lru_actions:export_data()
		}
	end
end

-- step(): Steps the enviroment
-- @param action_index: The index of the action to take (1-based)
-- @param info?: Any extra info / data
-- @return Types.step_return
function rl_env:step(action_index: number, info: any?): Types.step_return
	local action = self.action_list[action_index]

	self:add_action(action, action_index, info)

	local score, done = self:on_step(action_index)

	return {
		done = done or self.done,
		info = self.info,
		observation = self:get_observation(),
		score = score or self.score
	}
end

-- export_data(): Export the data
-- @return Types.rl_env_exported_data
function rl_env:export_data(): Types.rl_env_exported_data
	return {
		score = self.score,
		done = self.done,
		action_map = self.action_map,
		current_action = self.current_action,
		info = self.info,
		exported_lru_data = self.lru_actions:export_data()
	}
end

return rl_env