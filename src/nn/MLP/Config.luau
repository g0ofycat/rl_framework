--!strict

local Functions = script.Parent.Parent.libs.Functions

local Optimizers = script.Parent.Parent.libs.Optimizers

return {
    activation_function = require(Functions.ReLU),
    loss_function = require(Functions.HuberLoss),
    optimizer_function = require(Optimizers.Adam.Adam_Init),
    classification_function = require(Functions.Softmax)
}