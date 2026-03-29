--!strict

return
    -- Softmax(): Softmax activation function
    -- @param inputs: Array of input values
    -- @return { number }: Array of probabilities
    function (inputs: { number }): { number }
        local max = math.max(table.unpack(inputs))

        local exp_sum = 0
        local outputs = {}

        for i = 1, #inputs do
            outputs[i] = math.exp(inputs[i] - max)
            exp_sum += outputs[i]
        end

        for i = 1, #outputs do
            outputs[i] /= exp_sum
        end

        return outputs
    end