--!strict

return {
    -- ReLU(): ReLU activation function
    -- @param x: Input value
    -- @return number: math.max(0, x)
    Activation = function(x: number): number
        return math.max(0, x)
    end,

    -- ReLUDerivative(): ReLU derivative
    -- @param y: Output of the ReLU function
    -- @return number: Derivative value
    Derivative = function(y: number): number
        if y > 0 then
            return 1
        end

        return 0
    end
}