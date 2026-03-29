--!strict

local MLP = {}

MLP.__index = MLP

--========================
-- // IMPORTS
--========================

local nn_serializer = require("../../serializer/nn_serializer")

local nn_config = require("./Config")

--========================
-- // TYPES
--========================

local Types = require("./Types")

export type MLP_constructor_type = typeof(setmetatable({} :: Types.MLP_type, MLP))

--========================
-- // CONSTRUCTOR
--========================

-- new(): Creates a new neural network
-- @param InputNodes: Number of input nodes
-- @param HiddenLayers: Array of hidden layer sizes
-- @param OutputNodes: Number of output nodes
-- @param Dropout_Rate?: Optional dropout rate
-- @param Seed?: Optional random seed
-- @param PretrainedWeights?: Optional pre-trained weights
-- @param Config?: Types.ConfigType?
-- @return MLP_constructor_type
function MLP.new(InputNodes: number, HiddenLayers: { number }, OutputNodes: number, Dropout_Rate: number?, Seed: number?, PretrainedWeights: Types.NN_Data?, Config: Types.ConfigType?): MLP_constructor_type
	if InputNodes <= 0 or #HiddenLayers <= 0 or OutputNodes <= 0 then
		error("new(): HiddenLayers must not be empty; InputNodes and OutputNodes must be positive")
	end

	local new_config = table.clone(nn_config)

	if Config then
		for k, v in Config do
			new_config[k] = v
		end
	end

	local self = {}

	self.inputNodes = InputNodes
	self.hiddenLayers = HiddenLayers
	self.outputNodes = OutputNodes

	self.dropout_rate = Dropout_Rate or 0

	self.random_generator = Seed and Random.new(Seed) or Random.new()

	self.optimizer_function = new_config.optimizer_function.new()

	self.weights = {}
	self.biases = {}

	self.optimizerStateWeights = {}
	self.optimizerStateBiases = {}

	self.dropoutMasks = {}

	self.config = new_config

	local numLayers = #HiddenLayers + 1

	local layerSizes = { InputNodes }

	for i = 1, #HiddenLayers do
		table.insert(layerSizes, HiddenLayers[i])
	end

	table.insert(layerSizes, OutputNodes)

	for layer = 1, numLayers do
		local fromSize = layerSizes[layer]
		local toSize = layerSizes[layer + 1]

		self.optimizerStateWeights[layer] = {}
		self.optimizerStateBiases[layer] = {}
		self.dropoutMasks[layer] = {}

		for from = 1, fromSize do
			self.optimizerStateWeights[layer][from] = {}

			for to = 1, toSize do
				self.optimizerStateWeights[layer][from][to] = { m = 0, v = 0 }
			end
		end

		for to = 1, toSize do
			self.optimizerStateBiases[layer][to] = { m = 0, v = 0 }
		end
	end

	local MLP_Object = setmetatable(self, MLP)

	MLP_Object:InitData(InputNodes, HiddenLayers, OutputNodes, PretrainedWeights)

	return MLP_Object
end

--========================
-- // PUBLIC API
--========================

-- ForwardPropagation(): Propagates input data through the network
-- @param Data: Input values
-- @param training: If true, dropout is applied
-- @return ({ { number } }, { { number } }, { number }): layerInputs, layerOutputs, layerOutputs[numLayers]
function MLP:ForwardPropagation(Data: { number }, training: boolean): ({ { number } }, { { number } }, { number })
	local numLayers = #self.hiddenLayers + 1
	local layerInputs = {}
	local layerOutputs = {}

	layerOutputs[0] = Data

	for layer = 1, numLayers do
		local prevOutputs = layerOutputs[layer - 1]
		local currentLayerSize = layer <= #self.hiddenLayers and self.hiddenLayers[layer] or self.outputNodes

		layerInputs[layer] = {}
		layerOutputs[layer] = {}

		for to = 1, currentLayerSize do
			local sum = 0

			for from = 1, #prevOutputs do
				sum += prevOutputs[from] * self.weights[layer][from][to]
			end

			layerInputs[layer][to] = sum + self.biases[layer][to]
		end

		if layer <= #self.hiddenLayers then
			for to = 1, currentLayerSize do
				layerOutputs[layer][to] = self.config.activation_function.Activation(layerInputs[layer][to])
			end

			table.clear(self.dropoutMasks[layer])

			if training then
				for to = 1, currentLayerSize do
					if self.random_generator:NextNumber(0, 1) < self.dropout_rate :: number then
						layerOutputs[layer][to] = 0
						self.dropoutMasks[layer][to] = 0
					else
						layerOutputs[layer][to] /= (1 - self.dropout_rate)
						self.dropoutMasks[layer][to] = 1 / (1 - self.dropout_rate)
					end
				end
			else
				for to = 1, currentLayerSize do
					self.dropoutMasks[layer][to] = 1
				end
			end
		else
			for to = 1, currentLayerSize do
				layerOutputs[layer][to] = layerInputs[layer][to]
			end
		end
	end

	return layerInputs, layerOutputs, layerOutputs[numLayers]
end

-- BackPropagation(): Adjusts weights / biases based on error
-- @param Data: Input data
-- @param Target: Expected outputs
-- @param LayerInputs: Pre-activation values for each layer
-- @param LayerOutputs: Post-activation values for each layer
-- @param Outputs: Final outputs
-- @param LearningRate: Adjustment rate
function MLP:BackPropagation(Data: { number }, Target: { number }, LayerInputs: { {number} }, LayerOutputs: { {number} }, Outputs: { number }, LearningRate: number): ()
	local self = self :: Types.MLP_type

	local maxGrad = 10
	local numLayers = #self.hiddenLayers + 1

	local layerGradients = {}

	layerGradients[numLayers] = {}

	for k = 1, self.outputNodes do
		layerGradients[numLayers][k] = math.clamp(Outputs[k] - Target[k], -maxGrad, maxGrad)
	end

	for layer = numLayers - 1, 1, -1 do
		local currentLayerSize = self.hiddenLayers[layer]
		layerGradients[layer] = {}

		for j = 1, currentLayerSize do
			local Error = 0

			for k = 1, #layerGradients[layer + 1] do
				Error += layerGradients[layer + 1][k] * self.weights[layer + 1][j][k]
			end

			layerGradients[layer][j] = math.clamp(
				Error * self.config.activation_function.Derivative(LayerInputs[layer][j]),
				-maxGrad,
				maxGrad
			) * self.dropoutMasks[layer][j]
		end
	end

	for layer = 1, numLayers do
		local prevOutputs = layer == 1 and Data or LayerOutputs[layer - 1]
		local currentLayerSize = layer <= #self.hiddenLayers and self.hiddenLayers[layer] or self.outputNodes

		for from = 1, #prevOutputs do
			for to = 1, currentLayerSize do
				local gradient = layerGradients[layer][to] * prevOutputs[from]
				local update = self.optimizer_function:Optimize(
					LearningRate,
					gradient,
					self.optimizerStateWeights[layer][from][to]
				)
				self.weights[layer][from][to] -= update
			end
		end

		for to = 1, currentLayerSize do
			local gradient = layerGradients[layer][to]
			local update = self.optimizer_function:Optimize(
				LearningRate,
				gradient,
				self.optimizerStateBiases[layer][to]
			)
			self.biases[layer][to] -= update
		end
	end
end

-- InitData(): Initializes weights and biases
-- @param InputNodes: Number of input nodes
-- @param HiddenLayers: Array of hidden layer sizes
-- @param OutputNodes: Number of output nodes
-- @param PretrainedData?: Optional pre-trained data
function MLP:InitData(InputNodes: number, HiddenLayers: { number }, OutputNodes: number, PretrainedData: Types.NN_Data?): ()
	if PretrainedData then
		self:ImportData(PretrainedData)
	else
		local layerSizes = { InputNodes }

		for i = 1, #HiddenLayers do
			table.insert(layerSizes, HiddenLayers[i])
		end

		table.insert(layerSizes, OutputNodes)

		local numLayers = #layerSizes - 1

		for layer = 1, numLayers do
			local fromSize = layerSizes[layer]
			local toSize = layerSizes[layer + 1]

			self.biases[layer] = {}

			self.config.weight_init_function(self.weights, fromSize, toSize, layer, self.random_generator)

			for to = 1, toSize do
				self.biases[layer][to] = self.random_generator:NextNumber(-0.05, 0.05)
			end
		end
	end
end

-- Predict(): Feeds input through the network and returns predictions
-- @param Data: Input values
-- @return { number }: Predicted outputs
function MLP:Predict(Data: { number }): { number }
	if #Data ~= self.inputNodes then
		error(string.format("Predict(): Data length is %d, expected %d input nodes", #Data, self.inputNodes))
	end

	local _, _, logits = self:ForwardPropagation(Data, false)

	return self.nn_config.classification_function(logits)
end

-- Train(): Runs forward + backward propagation to train the network
-- @param Data: Input values
-- @param Target: Expected values
-- @param LearningRate: Learning rate (clamped [0.0001, 1])
-- @return (number, { number }): Cost, Logits
function MLP:Train(Data: { number }, Target: { number }, LearningRate: number): (number, { number })
	if #Data ~= self.inputNodes then
		error(string.format("Train(): Data length is %d, expected %d input nodes", #Data, self.inputNodes))
	end

	if #Target ~= self.outputNodes then
		error(string.format("Train(): Target length is %d, expected %d output nodes", #Target, self.outputNodes))
	end

	if LearningRate < 0.0001 or LearningRate > 1 then
		warn(string.format("Train(): LearningRate %.4f is outside recommended range [0.0001, 1], clamping", LearningRate))

		LearningRate = math.clamp(LearningRate, 0.0001, 1)
	end

	local hiddenInputs, hiddenOutputs, logits = self:ForwardPropagation(Data, true)

	local cost = self.nn_config.loss_function(logits, Target)

	self:BackPropagation(Data, Target, hiddenInputs, hiddenOutputs, logits, LearningRate)

	return cost, logits
end

-- SoftUpdate(): Soft updates the network parameters
-- @param targetMLP: The target network to update
-- @param tau: The soft update parameter
function MLP:SoftUpdate(targetMLP: MLP_constructor_type, tau: number): ()
	local numLayers = #self.hiddenLayers + 1

	for layer = 1, numLayers do
		local layerSizePrev = layer == 1 and self.inputNodes or self.hiddenLayers[layer - 1]
		local layerSizeCurrent = layer <= #self.hiddenLayers and self.hiddenLayers[layer] or self.outputNodes

		for from = 1, layerSizePrev do
			for to = 1, layerSizeCurrent do
				targetMLP.weights[layer][from][to] = 
					tau * self.weights[layer][from][to] + (1 - tau) * targetMLP.weights[layer][from][to]
			end
		end

		for to = 1, layerSizeCurrent do
			targetMLP.biases[layer][to] = 
				tau * self.biases[layer][to] + (1 - tau) * targetMLP.biases[layer][to]
		end
	end
end

--========================
-- // UTILITY API
--========================

-- ImportData(): Loads pre-trained weights and biases
-- @param NN_Data: Types.NN_Data
function MLP:ImportData(NN_Data: Types.NN_Data): ()
	self.weights = NN_Data.weights
	self.biases = NN_Data.biases
	self.hiddenLayers = NN_Data.hiddenLayers
end

-- ExportData(): Returns the weights and biases of the network
-- @return Types.NN_Data
function MLP:ExportData(): Types.NN_Data
	return {
		weights = self.weights,
		biases = self.biases,
		hiddenLayers = self.hiddenLayers
	}
end

-- SaveNN(): Returns the weights and biases as a string
-- @return string: Encoded string of weights and biases
function MLP:SaveNN(): string
	return nn_serializer.serialize(self:ExportData())
end

-- LoadNN(): Loads the weights and biases from the string
-- @param data: The encoded string of weights and biases
function MLP:LoadNN(data: string): ()
	return self:ImportData(nn_serializer.deserialize(data))
end

return MLP