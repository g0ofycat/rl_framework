--!strict

local Adam = {}

Adam.__index = Adam

--=========================
-- // TYPES
--=========================

local Types = require("./Types")

export type adam_constructor_type = typeof(setmetatable({} :: Types.AdamType, Adam))

--=========================
-- // CONSTRUCTOR
--=========================

-- new(): Creates a new Adam optimizer instance
-- @return adam_constructor_type
function Adam.new(): adam_constructor_type
    local self = setmetatable({
        First_Moment_Decay = 0.9,
        Second_Moment_Decay = 0.999,
        Epsilon = 1e-8,
        TimeStep = 0
    }, Adam)

    return self
end

--=========================
-- // OPTIMIZER API
--=========================

-- Optimize(): Adam optimizer
-- @param learning_rate: number
-- @param gradients: number
-- @param state: { m: number, v: number }
-- @return number: Updated parameter value
function Adam:Optimize(
    learning_rate: number,
    gradients: number,
    state: { m: number, v: number }
): number
    local self = self :: Types.AdamType

    self.TimeStep += 1

    local m = state.m
    local v = state.v

    m = self.First_Moment_Decay * m + (1 - self.First_Moment_Decay) * gradients
    v = self.Second_Moment_Decay * v + (1 - self.Second_Moment_Decay) * (gradients * gradients)

    local m_hat = m / (1 - self.First_Moment_Decay ^ self.TimeStep)
    local v_hat = v / (1 - self.Second_Moment_Decay ^ self.TimeStep)

    local update = learning_rate * (m_hat / (math.sqrt(v_hat) + self.Epsilon))

    state.m = m
    state.v = v

    return update
end

return Adam