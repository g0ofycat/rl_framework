--!strict

return {
    -- Sigmoid(): Sigmoid activation function
    -- @param x: Input value
    -- @return number: Output between 0 and 1
    Activation = function(x: number): number
        if x >= 0 then
            return 1 / (1 + math.exp(-x))
        else
            return math.exp(x) / (1 + math.exp(x))
        end
    end,

    -- SigmoidDerivative(): Sigmoid derivative
    -- @param y: Output of the Sigmoid function
    -- @return number: Derivative value
    Derivative = function(y: number): number
        return y * (1 - y)
    end
}