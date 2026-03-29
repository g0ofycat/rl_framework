--!strict

--========================
-- // REFERENCES
--========================

local Functions = script.Parent.Parent.libs.Functions
local Optimizers = script.Parent.Parent.libs.Optimizers
local WeightInit = script.Parent.Parent.libs.WeightInit

--========================
-- // RETURN
--========================

return {
    activation_function = require(Functions.ReLU),
    loss_function = require(Functions.HuberLoss),
    optimizer_function = require(Optimizers.Adam.Adam_Init),
    classification_function = require(Functions.Softmax),

    weight_init_function = require(WeightInit.XavierUniform)
}