--!strict

return {
    -- GeLU(): GeLU activation function (sigmoid-based approximation)
    -- @param x: Input value
    -- @return number: Output
    Activation = function(x: number): number
        return 0.5 * x * (1 + (1 / (1 + math.exp(-math.sqrt(2 / math.pi) * (x + 0.044715 * x ^ 3)))))
    end,

    -- GeLUDerivative(): GeLU derivative
    -- @param x: Input value
    -- @return number: Derivative value
    Derivative = function(x: number): number
        local a = math.sqrt(2 / math.pi) * (x + 0.044715 * x^3)
        local s = 1 / (1 + math.exp(-a))
        local a_prime = math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x^2)

        return 0.5 * (1 + s) + 0.5 * x * s * (1 - s) * a_prime
    end
}