--!strict

return {
    -- ReLU(): ReLU activation function
    -- @param x: Input value
    -- @return number: math.max(0, x)
    Activation = function(x: number): number
        return math.max(0, x)
    end,

    -- ReLUDerivative(): ReLU derivative
    -- @param x: Input value
    -- @return number: 1 if x > 0, 0 otherwise
    Derivative = function(x: number): number
        if x > 0 then
            return 1
        else
            return 0
        end
    end
}