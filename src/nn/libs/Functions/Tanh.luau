--!strict

return {
    -- Tanh(): Tanh activation function
    -- @param x: Input value
    -- @return number: Tanh of x
    Activation = function(x: number): number
        return math.tanh(x)
    end,

    -- TanhDerivative(): Tanh derivative
    -- @param y: Output of Tanh function
    -- @return number: Derivative value
    Derivative = function(y: number): number
        return 1 - y * y
    end
}