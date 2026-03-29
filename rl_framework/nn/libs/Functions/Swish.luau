--!strict

return {
    -- Swish(): Swish activation function
    -- @param x: Input value
    -- @return number: Output
    Activation = function(x: number): number
        if x >= 0 then
            return x * (1 / (1 + math.exp(-x)))
        else
            return x * (math.exp(x) / (1 + math.exp(x)))
        end
    end,

    -- SwishDerivative(): Swish derivative
    -- @param x: Input value
    -- @return number: Derivative value
    Derivative = function(x: number): number
        local s

        if x >= 0 then
            s = 1 / (1 + math.exp(-x))
        else
            s = math.exp(x) / (1 + math.exp(x))
        end

        return s + x * s * (1 - s)
    end
}