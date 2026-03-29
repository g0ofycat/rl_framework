--!strict

return
    -- MSE(): Mean Squared Error
    -- @param Outputs: Predicted values
    -- @param Expected: Expected values
    -- @return error: The average error
    function (Outputs: { number }, Expected: { number }): number
        if #Outputs ~= #Expected then
            error("MSE(): Actuals and predictions tables must have the same number of elements")
        end

        local sum_squared_error = 0

        for i = 1, #Expected do
            local error = Expected[i] - Outputs[i]

            sum_squared_error = sum_squared_error + math.pow(error, 2)
        end

        local mean_squared_error = sum_squared_error / #Outputs

        return mean_squared_error
    end