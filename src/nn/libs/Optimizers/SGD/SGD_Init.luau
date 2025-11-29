--!strict

local SGD = {}

SGD.__index = SGD

--=========================
-- // TYPES
--=========================

local Types = require("./Types")

export type sgd_constructor_type = typeof(setmetatable({} :: Types.SGDType, SGD))

--=========================
-- // CONSTRUCTOR
--=========================

-- new(): Creates a new SGD optimizer instance
-- @return sgd_constructor_type
function SGD.new(): sgd_constructor_type
    local self = setmetatable({}, SGD)

    return self
end

--=========================
-- // OPTIMIZER API
--=========================

-- Optimize(): SGD optimizer
-- @param learning_rate: number
-- @param gradients: number
-- @param _?: any? (For compatibility with other optimizers)
-- @return number: Updated parameter value
function SGD:Optimize(
    learning_rate: number,
    gradients: number,
    _: any?
): number
    return learning_rate * gradients
end

return SGD