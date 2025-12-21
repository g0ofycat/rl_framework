--!strict

local nn_serializer = {}

--=========================
-- // SERVICES
--=========================

local HttpService = game:GetService("HttpService")

--=========================
-- // MODULES
--=========================

local MLP_Types = require("../nn/MLP/Types")

--=========================
-- // PRIVATE API
--=========================

-- delta_compress(): Compresses a set of weights and biases into a single tensor
-- @param data: The neural network data to compress
-- @return MLP_Types.NN_Data: Delta encoded neural network data
local function delta_compress(data: MLP_Types.NN_Data): MLP_Types.NN_Data
    local compressed_weights = {}

    for layer_idx, layer in data.weights do
        compressed_weights[layer_idx] = {}

        for neuron_idx, neuron_weights in layer do
            compressed_weights[layer_idx][neuron_idx] = {}

            compressed_weights[layer_idx][neuron_idx][1] = neuron_weights[1]

            for weight_idx = 2, #neuron_weights do
                local delta = neuron_weights[weight_idx] - neuron_weights[weight_idx - 1]

                compressed_weights[layer_idx][neuron_idx][weight_idx] = delta
            end
        end
    end

    local compressed_biases = {}

    for layer_idx, layer_biases in data.biases do
        compressed_biases[layer_idx] = {}

        compressed_biases[layer_idx][1] = layer_biases[1]

        for bias_idx = 2, #layer_biases do
            local delta = layer_biases[bias_idx] - layer_biases[bias_idx - 1]

            compressed_biases[layer_idx][bias_idx] = delta
        end
    end

    return {
        weights = compressed_weights,
        biases = compressed_biases,
        hiddenLayers = data.hiddenLayers
    }
end

-- delta_decompress(): Decompresses a tensor back into a set of weights and biases
-- @param compressed_data: The compressed tensor as a JSON string
-- @return MLP_Types.NN_Data: Delta decoded neural network data
local function delta_decompress(compressed_data: MLP_Types.NN_Data): MLP_Types.NN_Data
    local decompressed_weights = {}

    for layer_idx, layer in compressed_data.weights do
        decompressed_weights[layer_idx] = {}

        for neuron_idx, neuron_deltas in layer do
            decompressed_weights[layer_idx][neuron_idx] = {}

            decompressed_weights[layer_idx][neuron_idx][1] = neuron_deltas[1]

            for weight_idx = 2, #neuron_deltas do
                local previous_weight = decompressed_weights[layer_idx][neuron_idx][weight_idx - 1]

                decompressed_weights[layer_idx][neuron_idx][weight_idx] = previous_weight + neuron_deltas[weight_idx]
            end
        end
    end

    local decompressed_biases = {}

    for layer_idx, layer_deltas in compressed_data.biases do
        decompressed_biases[layer_idx] = {}

        decompressed_biases[layer_idx][1] = layer_deltas[1]

        for bias_idx = 2, #layer_deltas do
            local previous_bias = decompressed_biases[layer_idx][bias_idx - 1]

            decompressed_biases[layer_idx][bias_idx] = previous_bias + layer_deltas[bias_idx]
        end
    end

    return {
        weights = decompressed_weights,
        biases = decompressed_biases,
        hiddenLayers = compressed_data.hiddenLayers
    }
end

--=========================
-- // PUBIC API
--=========================

-- serialize(): Compress and serialize into a JSON string
-- @param data: The neural network data to serialize
-- @return string: The serialized JSON string
function nn_serializer.serialize(data: MLP_Types.NN_Data): string
    local compressed_data = delta_compress(data)

    return HttpService:JSONEncode(compressed_data)
end

-- deserialize(): Deserialize and decompress from a JSON string
-- @param compressed_data: The JSON string to deserialize and decompress
-- @return MLP_Types.NN_Data: The decompressed neural network data
function nn_serializer.deserialize(compressed_data: string): MLP_Types.NN_Data
    local deserialized_data = HttpService:JSONDecode(compressed_data)

    return delta_decompress(deserialized_data)
end

return nn_serializer