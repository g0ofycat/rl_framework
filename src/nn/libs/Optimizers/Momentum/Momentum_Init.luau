--!strict

local Momentum = {}

Momentum.__index = Momentum

--=========================
-- // TYPES
--=========================

local Types = require("./Types")

export type momentum_constructor_type = typeof(setmetatable({} :: Types.MomentumType, Momentum))

--=========================
-- // CONSTRUCTOR
--=========================

-- new(): Creates a new Momentum optimizer instance
-- @return momentum_constructor_type
function Momentum.new(): momentum_constructor_type
    local self = setmetatable({
        Beta = 0.9
    }, Momentum)

    return self
end

--=========================
-- // OPTIMIZER API
--=========================

-- Optimize(): Momentum optimizer
-- @param learning_rate: number
-- @param gradients: number
-- @param state: { m: number, v: number }
-- @return number: Updated parameter value
function Momentum:Optimize(
    learning_rate: number,
    gradients: number,
    state: { m: number, v: number }
): number
    local self = self :: Types.MomentumType

   state.m = self.Beta * state.m + (1 - self.Beta) * gradients

    return learning_rate * state.m
end

return Momentum