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
    -- @param y: Output of the LeakyReLU function
    -- @return number: Derivative value
    Derivative = function(y: number): number
        local alpha = 0.01

        if y > 0 then
            return 1
        else
            return alpha
        end
    end
}