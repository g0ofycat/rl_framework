--!strict
--!native
--!optimize 2

local FFN = {}

FFN.__index = FFN

--========================
-- // IMPORTS
--========================

local FunctionLib = require("../libs/FunctionLib")

local Optimizers = require("../libs/Optimizers/Optimizers_Init")

local ts_database = require("../../db/ts_database/ts_database_init")

local config = require("../../config")

--========================
-- // TYPES
--========================

local Types = require("./Types")

export type FFN_constructor_type = typeof(setmetatable({} :: Types.FFN_type, FFN))

--========================
-- // DB INIT
--========================

ts_database.set_api_key(config.strings.db_api_key)

ts_database.load_version("FFN_Data")

--========================
-- // CONSTRUCTOR
--========================

-- new(): Creates a new neural network
-- @param InputNodes: Number of input nodes
-- @param HiddenNodes: Number of hidden nodes
-- @param OutputNodes: Number of output nodes
-- @param Dropout_Rate?: Optional dropout rate
-- @param Seed?: Optional random seed
-- @param PretrainedWeights?: Optional pre-trained weights
-- @return FFN_constructor_type
function FFN.new(InputNodes: number, HiddenNodes: number, OutputNodes: number, Dropout_Rate: number?, Seed: number?, PretrainedWeights: Types.NN_Data?): FFN_constructor_type
	if InputNodes <= 0 or HiddenNodes <= 0 or OutputNodes <= 0 then
		error("New(): InputNodes, HiddenNodes, and OutputNodes must be positive")
	end

	local self = {}

	self.inputNodes = InputNodes
	self.hiddenNodes = HiddenNodes
	self.outputNodes = OutputNodes

	self.dropout_rate = Dropout_Rate or 0

	self.timeStep = 1

	self.random_generator = Seed and Random.new(Seed) or Random.new()

	self.weightsIH = {}
	self.weightsHO = {}
	self.biasH = {}
	self.biasO = {}

	self.adamStateIH = {}
	self.adamStateHO = {}
	self.adamStateBiasH = {}
	self.adamStateBiasO = {}

	for i = 1, InputNodes do
		self.adamStateIH[i] = {}

		for j = 1, HiddenNodes do
			self.adamStateIH[i][j] = { m = 0, v = 0 }
		end
	end

	for j = 1, HiddenNodes do
		self.adamStateHO[j] = {}

		for k = 1, OutputNodes do
			self.adamStateHO[j][k] = { m = 0, v = 0 }
		end

		self.adamStateBiasH[j] = { m = 0, v = 0 }
	end

	for k = 1, OutputNodes do
		self.adamStateBiasO[k] = { m = 0, v = 0 }
	end

	self.dropoutMask = {}

	local FFN_Object = setmetatable(self, FFN)

	FFN_Object:InitData(InputNodes, HiddenNodes, OutputNodes, PretrainedWeights)

	return FFN_Object
end

--========================
-- // PUBLIC API
--========================

-- ForwardPropagation(): Propagates input data through the network
-- @param Data: Input values
-- @param training: If true, dropout is applied
-- @return ({ number }, { number }, { number }): hiddenInputs, hiddenOutputs, outputInputs
function FFN:ForwardPropagation(Data: { number }, training: boolean): ({ number }, { number }, { number })
	local hiddenInputs, hiddenOutputs = {}, {}

	table.clear(self.dropoutMask)

	for j = 1, self.hiddenNodes do
		local sum = 0

		for i = 1, self.inputNodes do
			sum += Data[i] * self.weightsIH[i][j]
		end

		hiddenInputs[j] = sum + self.biasH[j]
		hiddenOutputs[j] = FunctionLib.LeakyReLU(hiddenInputs[j])
	end

	if training then
		for j = 1, self.hiddenNodes do
			if self.random_generator:NextNumber(0, 1) < self.dropout_rate :: number then
				hiddenOutputs[j] = 0
				self.dropoutMask[j] = 0
			else
				hiddenOutputs[j] /= (1 - self.dropout_rate)
				self.dropoutMask[j] = 1 / (1 - self.dropout_rate)
			end
		end
	else
		for j = 1, self.hiddenNodes do
			self.dropoutMask[j] = 1
		end
	end

	local outputInputs = {}

	for k = 1, self.outputNodes do
		local sum = 0

		for j = 1, self.hiddenNodes do
			sum += hiddenOutputs[j] * self.weightsHO[j][k]
		end

		outputInputs[k] = sum + self.biasO[k]
	end

	return hiddenInputs, hiddenOutputs, outputInputs
end

-- BackPropagation(): Adjusts weights / biases based on error
-- @param Data: Input data
-- @param Target: Expected outputs
-- @param HiddenInputs: Hidden layer inputs
-- @param HiddenOutputs: Hidden layer outputs
-- @param Outputs: Final outputs
-- @param LearningRate: Adjustment rate
function FFN:BackPropagation(Data: { number }, Target: { number }, HiddenInputs: { number }, HiddenOutputs: { number }, Outputs: { number }, LearningRate: number): ()
	local self = self :: Types.FFN_type
	local oldHO = {}
	local maxGrad = 10

	for j = 1, self.hiddenNodes do
		oldHO[j] = {}

		for k = 1, self.outputNodes do
			oldHO[j][k] = self.weightsHO[j][k]
		end
	end

	local outputGradients = {}

	for k = 1, self.outputNodes do
		outputGradients[k] = math.clamp(Outputs[k] - Target[k], -maxGrad, maxGrad)
	end

	local hiddenGradients = {}

	for j = 1, self.hiddenNodes do
		local Error = 0

		for k = 1, self.outputNodes do
			Error += outputGradients[k] * oldHO[j][k]
		end

		hiddenGradients[j] = math.clamp(Error * FunctionLib.LeakyReLUDerivative(HiddenInputs[j]), -maxGrad, maxGrad) * self.dropoutMask[j]
	end

	for k = 1, self.outputNodes do
		for j = 1, self.hiddenNodes do
			local gradients = outputGradients[k] * HiddenOutputs[j]

			local update = Optimizers.Adam(LearningRate, gradients, self.timeStep, self.adamStateHO[j][k])

			self.weightsHO[j][k] -= update
		end

		local gradients = outputGradients[k]

		local update = Optimizers.Adam(LearningRate, gradients, self.timeStep, self.adamStateBiasO[k])

		self.biasO[k] -= update
	end

	for j = 1, self.hiddenNodes do
		for i = 1, self.inputNodes do
			local gradients = hiddenGradients[j] * Data[i]

			local update = Optimizers.Adam(LearningRate, gradients, self.timeStep, self.adamStateIH[i][j])

			self.weightsIH[i][j] -= update
		end

		local gradients = hiddenGradients[j]

		local update = Optimizers.Adam(LearningRate, gradients, self.timeStep, self.adamStateBiasH[j])

		self.biasH[j] -= update
	end

	self.timeStep += 1
end

-- InitData(): Initializes weights and biases
-- @param InputNodes: Number of input nodes
-- @param HiddenNodes: Number of hidden nodes
-- @param OutputNodes: Number of output nodes
-- @param PretrainedData?: Optional pre-trained data
function FFN:InitData(InputNodes: number, HiddenNodes: number, OutputNodes: number, PretrainedData: Types.NN_Data?): ()
	if PretrainedData then
		self:ImportData(PretrainedData)
	else
		for i = 1, InputNodes do
			self.weightsIH[i] = {}

			for j = 1, HiddenNodes do
				local scaleIH = math.sqrt(6 / (InputNodes + HiddenNodes))
				self.weightsIH[i][j] = self.random_generator:NextNumber(-scaleIH, scaleIH)
			end
		end

		for j = 1, HiddenNodes do
			self.weightsHO[j] = {}

			for k = 1, OutputNodes do
				local scaleHO = math.sqrt(6 / (HiddenNodes + OutputNodes))
				self.weightsHO[j][k] = self.random_generator:NextNumber(-scaleHO, scaleHO)
			end
		end

		for j = 1, HiddenNodes do
			self.biasH[j] = self.random_generator:NextNumber(-0.05, 0.05)
		end

		for k = 1, OutputNodes do
			self.biasO[k] = self.random_generator:NextNumber(-0.05, 0.05)
		end
	end
end

-- Predict(): Feeds input through the network and returns predictions (Softmaxxed)
-- @param Data: Input values
-- @param Temperature: Temperature for sampling
-- @return { number }: Predicted outputs
function FFN:Predict(Data: { number }, Temperature: number): { number }
	if #Data ~= self.inputNodes then
		error(string.format("Predict(): Data length is %d, expected %d input nodes", #Data, self.inputNodes))
	end

	local _, _, logits = self:ForwardPropagation(Data, false)

	return FunctionLib.Softmax(FunctionLib.Temperature(logits, Temperature))
end

-- Train(): Runs forward + backward propagation to train the network
-- @param Data: Input values
-- @param Target: Expected values
-- @param LearningRate: Learning rate (clamped [0.0001, 1])
-- @return (number, { number }): Predicted outputs
function FFN:Train(Data: { number }, Target: { number }, LearningRate: number): (number, { number })
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

	local cost = FunctionLib.HuberLoss(logits, Target)

	self:BackPropagation(Data, Target, hiddenInputs, hiddenOutputs, logits, LearningRate)

	return cost, logits
end

-- SoftUpdate(): Soft updates the network parameters of the network
-- @param targetFFN: The target network to update
-- @param tau: The soft update parameter
function FFN:SoftUpdate(targetFFN: FFN_constructor_type, tau: number): ()
	for i = 1, self.inputNodes do
		for j = 1, self.hiddenNodes do
			targetFFN.weightsIH[i][j] =
				tau * self.weightsIH[i][j] + (1 - tau) * targetFFN.weightsIH[i][j]
		end
	end

	for j = 1, self.hiddenNodes do
		for k = 1, self.outputNodes do
			targetFFN.weightsHO[j][k] =
				tau * self.weightsHO[j][k] + (1 - tau) * targetFFN.weightsHO[j][k]
		end
		targetFFN.biasH[j] = tau * self.biasH[j] + (1 - tau) * targetFFN.biasH[j]
	end

	for k = 1, self.outputNodes do
		targetFFN.biasO[k] = tau * self.biasO[k] + (1 - tau) * targetFFN.biasO[k]
	end
end

--========================
-- // DB OPERATIONS
--========================

-- ImportData(): Loads pre-trained weights and biases
-- @param NN_Data: Types.NN_Data
function FFN:ImportData(NN_Data: Types.NN_Data): ()
	self.weightsIH = NN_Data.weightsIH
	self.weightsHO = NN_Data.weightsHO

	self.biasH = NN_Data.biasH
	self.biasO = NN_Data.biasO
end

-- ExportData(): Returns the weights and biases of the network
-- @return Types.NN_Data
function FFN:ExportData(): Types.NN_Data
	return {
		weightsIH = self.weightsIH,
		weightsHO = self.weightsHO,

		biasH = self.biasH,
		biasO = self.biasO
	}
end

-- SaveToDB(): Saves the weights and biases to the database
-- @param name?: Custom name, if any
-- @return id: The ID of the inserted data
function FFN:SaveToDB(name: any?): number
	local id = ts_database.insert({ name = name, data = self:ExportData() })

	return id
end

-- LoadFromDB(): Loads the weights and biases from the database
-- @param id: The ID of the data to load
function FFN:LoadFromDB(id: number): ()
	local data = ts_database.get(id)

	self:ImportData(data)
end

return FFN