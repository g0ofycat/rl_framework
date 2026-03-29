--!strict

return {
    -- LeakyReLU(): LeakyReLU activation function
    -- @param x: Input value
    -- @return number: x if positive, alpha * x otherwise
    Activation = function(x: number): number
        local alpha = 0.01

        if x > 0 then
            return x
        else
            return alpha * x
        end
    end,

    -- LeakyReLUDerivative(): LeakyReLU derivative
    -- @param x: Input value
    -- @return number: 1 if x > 0, alpha otherwise
    Derivative = function(x: number): number
        local alpha = 0.01

        if x > 0 then
            return 1
        else
            return alpha
        end
    end
}