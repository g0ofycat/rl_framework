# rl_framework

welcome to another project that i made that has absolutley 0 use case because who even trains models on roblox

reinforcement learning framework wip, i'll add more types of nn's later. mainly rnn's

## api

agents can be created using **agent_instance.new()**

**function signature:**

```lua
-- new(): Create a new agent
-- @param state_size: The size of the state
-- @param action_size: The size of the action
-- @param action_map: Dictionary map for action to reward
-- @param lru_cache_capacity: Cache capacity for LRU
-- @param callbacks?: Callbacks for environment events
-- @param env_variables?: Environment variables
-- @param config?: The configuration
-- @return agent_instance_constructor_type
function agent_instance.new(state_size: number, action_size: number, action_map: { [any]: number }, lru_cache_capacity: number, callbacks: Types.callback_table?, env_variables: any?, config: Types.ConfigType?): agent_instance_constructor_type
```

**example:**

```lua
local new_agent = agent_instance.new(0, 0,
{}, 0, {

	on_reset = function(self)

	end,

	on_step = function(self, action_index)

	end,

    on_action = function(self, action_index)

    end,

	observation_overwrite = function(self)

	end
}, {}, {}, {})
```

**state_size -** *Amount of input neurons*

**action_size -** *Amount of output neurons*

**action_map -** *Map of an action to the reward value (Usually 1 or 0, you do reward logic in on_step)*

**lru_cache_capacity -** *Array size index limit for the memory, uses a LRU Cache*

**callbacks -** *Main logic of the enviroment and agent interaction, see "setting up enviroment" section*

**env_variables -** *Usually a hashmap of variables that should be exposed only to the enviroment, should be used in the callback functions*

**config -** *Custom configuration to override default_config.luau (Can be Partial)*

**nn_config -** *Custom configuration to override MLP/Config.luau (Can be Partial)*

## setting up enviroment:

when setting up an enviroment, you have these **4** callbacks:

```lua
on_reset = function(self)
    --[[
        - Triggers every episode change

        - @param self: The enviroment
    ]]
end,

on_step = function(self, action_index)
    --[[
        - Triggers every step

        - @param self: The enviroment
        - @param action_index: The index of the action that has been chosen
    ]]
end,

on_action = function(self, action_index)
    --[[
        - Triggers every action pick

        - @param self: The enviroment
        - @param action_index: The index of the action that has been chosen
    ]]
end,

observation_overwrite = function(self)
    --[[
        - Data to be returned and observed

        - @param self: The enviroment

        - @return observation_type
    ]]
end
```

*note: to get an action from an index, use **agent:get_env_variables()** outside of the callback scopes. Inside of the callback scopes, you can just do **self.env_variables***

## training & inferencing:

### training:

**to train, use:**

```lua
-- run_episodes(): Runs multiple episodes
-- @param num_episodes: The number of episodes to run
-- @param episode_callback?: The function to call after each episode (Current Episode, Reward)
-- @param step_callback?: The function to call each step (observation)
function agent_instance:run_episodes(
	num_episodes: number,
	episode_callback: ((episode: number, reward: number) -> ())?,
	step_callback: ((observation: { number }) -> ())?
): ()
```

parameters are self explanitory, use this method after you've defined your enviroment and agent

### inferencing:

**to inference loop, use:**

```lua
-- create_inference_thread(): Creates a new inference thread
-- @param get_state: Function to return the current state
-- @param on_action: Callback function to call when an action is selected (action_index)
-- @return thread: The thread loop
function agent_instance:create_inference_thread(get_state: () -> { number }, on_action: (action_index: number) -> ()): thread
```

**to inference once, use:**

```lua
-- get_best_action(): Get the best action (Forward Pass)
-- @param state: The current state
-- @return number: Best Action
function agent_instance:get_best_action(state: { number }): number
```

## settings:

rl framework uses DQN and Epsilon-Greedy (With Decay); settings are self explanatory

**example settings for training (found in agent/agent_instance/default_config.luau):**

```lua
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
```