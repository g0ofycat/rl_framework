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
local Base64 = require("./libs/Base64")

--=========================
-- // CONSTANTS
--=========================

local Q_LEVELS = 255

--=========================
-- // PRIVATE API
--=========================

-- find_min_max(): Finds global min and max across weights and biases
-- @param data: The neural network data
-- @return number, number: min and max values
local function find_min_max(data: MLP_Types.NN_Data): (number, number)
    local min_val = math.huge
    local max_val = -math.huge

    for _, layer in data.weights do
        for _, neuron in layer do
            for _, value in neuron do
                min_val = math.min(min_val, value)
                max_val = math.max(max_val, value)
            end
        end
    end

    for _, layer in data.biases do
        for _, value in layer do
            min_val = math.min(min_val, value)
            max_val = math.max(max_val, value)
        end
    end

    return min_val, max_val
end

-- quantize(): Converts a float into an integer representation
-- @param value: The float value
-- @param min_val: Global minimum
-- @param scale: Quantization scale
-- @return number: Quantized integer
local function quantize(value: number, min_val: number, scale: number): number
    return ((value - min_val) / scale + 0.5) // 1
end

-- dequantize(): Converts an integer back into a float
-- @param q: Quantized integer
-- @param min_val: Global minimum
-- @param scale: Quantization scale
-- @return number: Dequantized float
local function dequantize(q: number, min_val: number, scale: number): number
    return q * scale + min_val
end

-- encode_weights(): Quantizes and base64 encodes weights
-- @param data: The neural network data
-- @param min_val: Global minimum
-- @param scale: Quantization scale
-- @return string: Base64 encoded weights
local function encode_weights(data: MLP_Types.NN_Data, min_val: number, scale: number): string
    local bytes = {}

    for _, layer in data.weights do
        for _, neuron in layer do
            for _, value in neuron do
                bytes[#bytes + 1] = string.char(quantize(value, min_val, scale))
            end
        end
    end

    return Base64.Encode(table.concat(bytes))
end

-- encode_biases(): Quantizes and base64 encodes biases
-- @param data: The neural network data
-- @param min_val: Global minimum
-- @param scale: Quantization scale
-- @return string: Base64 encoded biases
local function encode_biases(data: MLP_Types.NN_Data, min_val: number, scale: number): string
    local bytes = {}

    for _, layer in data.biases do
        for _, value in layer do
            bytes[#bytes + 1] = string.char(quantize(value, min_val, scale))
        end
    end

    return Base64.Encode(table.concat(bytes))
end

-- decode_weights(): Decodes and dequantizes weights
-- @param encoded: Base64 encoded weights
-- @param layerSizes: The full network layer sizes including input and output
-- @param min_val: Global minimum
-- @param scale: Quantization scale
-- @return MLP_Types.NN_Weights: Decoded weights
local function decode_weights(encoded: string, layerSizes: { number }, min_val: number, scale: number): MLP_Types.NN_Weights
    local raw = Base64.Decode(encoded)
    local weights: MLP_Types.NN_Weights = {}
    local byte_idx = 1

    for layer_idx = 1, #layerSizes - 1 do
        weights[layer_idx] = {}
        local input_count = layerSizes[layer_idx]
        local neuron_count = layerSizes[layer_idx + 1]

        for from = 1, input_count do
            weights[layer_idx][from] = {}

            for to = 1, neuron_count do
                weights[layer_idx][from][to] = dequantize(raw:byte(byte_idx), min_val, scale)
                byte_idx += 1
            end
        end
    end

    return weights
end

-- decode_biases(): Decodes and dequantizes biases
-- @param encoded: Base64 encoded biases
-- @param layerSizes: The full network layer sizes including input and output
-- @param min_val: Global minimum
-- @param scale: Quantization scale
-- @return MLP_Types.NN_Biases: Decoded biases
local function decode_biases(encoded: string, layerSizes: { number }, min_val: number, scale: number): MLP_Types.NN_Biases
    local raw = Base64.Decode(encoded)
    local biases: MLP_Types.NN_Biases = {}
    local byte_idx = 1

    for layer_idx = 2, #layerSizes do
        biases[layer_idx - 1] = {}
        local neuron_count = layerSizes[layer_idx]

        for b = 1, neuron_count do
            biases[layer_idx - 1][b] = dequantize(raw:byte(byte_idx), min_val, scale)
            byte_idx += 1
        end
    end

    return biases
end

--=========================
-- // PUBIC API
--=========================

-- serialize(): Compress and serialize into a JSON string
-- @param data: The neural network data to serialize
-- @return string: The serialized JSON string
function nn_serializer.serialize(data: MLP_Types.NN_Data): string
	local min_val, max_val = find_min_max(data)

	local scale = (max_val - min_val) / Q_LEVELS
	if scale == 0 then scale = 1 end

	return HttpService:JSONEncode({
		inputNodes = data.inputNodes,
		hiddenLayers = data.hiddenLayers,
        layerSizes = data.layerSizes,
		outputNodes = data.outputNodes,

		weights = encode_weights(data, min_val, scale),
		biases = encode_biases(data, min_val, scale),

		min = min_val,
		scale = scale
	})
end

-- deserialize(): Deserialize and decompress from a JSON string
-- @param compressed_data: The JSON string to deserialize and decompress
-- @return MLP_Types.NN_Data: The decompressed neural network data
function nn_serializer.deserialize(compressed_data: string): MLP_Types.NN_Data
	local d = HttpService:JSONDecode(compressed_data)

	local min_val = d.min :: number
	local scale = d.scale :: number

	local layerSizes = d.layerSizes :: { number }

	return {
		inputNodes = d.inputNodes,
		hiddenLayers = d.hiddenLayers,
        layerSizes = layerSizes,
		outputNodes = d.outputNodes,

		weights = decode_weights(d.weights, layerSizes, min_val, scale),
		biases = decode_biases(d.biases, layerSizes, min_val, scale),

		min = min_val,
		scale = scale
	}
end

return nn_serializer