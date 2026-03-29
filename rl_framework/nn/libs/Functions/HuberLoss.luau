--!strict

return
    -- HuberLoss(): Huber Loss
    -- @param Outputs: Predicted values
    -- @param Expected: Expected values
    -- @return number: The average error
    function (Outputs: { number }, Expected: { number }): number
        local delta = 1.0

        local totalLoss = 0

        for i = 1, #Outputs do
            local Error = Outputs[i] - Expected[i]
            local absError = math.abs(Error)

            if absError <= delta then
                totalLoss += 0.5 * Error * Error
            else
                totalLoss += delta * (absError - 0.5 * delta)
            end
        end

        return totalLoss / #Outputs
    end