--!strict

local FunctionLib = {}

--=========================
-- // MAIN FUNCS
--=========================

-- Softmax(): Softmax activation function
-- @param inputs { number }: Array of input values
-- @return { number }: Array of probabilities
function FunctionLib.Softmax(inputs: { number }): { number }
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

-- CrossEntropyLoss(): Cross Entropy Loss
-- @param Outputs: Predicted values
-- @param Expected: Expected values
-- @return cost: The average error
function FunctionLib.CrossEntropyLoss(Outputs: { number }, Expected: { number }): number
	local sum = 0

	for i = 1, #Outputs do
		local pred = math.clamp(Outputs[i], 1e-7, 1 - 1e-7)

		sum += -Expected[i] * math.log(pred)
	end

	return sum / #Outputs
end

-- MSE(): Mean Squared Error
-- @param Outputs: Predicted values
-- @param Expected: Expected values
-- @return error: The average error
function FunctionLib.MSE(Outputs: { number }, Expected: { number }): number
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

-- HuberLoss(): Huber Loss
-- @param Outputs: Predicted values
-- @param Expected: Expected values
-- @return number: The average error
function FunctionLib.HuberLoss(Outputs: { number }, Expected: { number }): number
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

--=========================
-- // SAMPLING
--=========================

-- Temperature(): Tempature sampling
-- @param Probabilities: Raw logits
-- @return { number }: Transformed logits
function FunctionLib.Temperature(logits: { number }, tempature: number): { number }
	if tempature <= 0 then
		return logits
	end

	local tempature_logits = {}

	for i, logit in logits do
		table.insert(tempature_logits, logit / tempature)
	end

	return tempature_logits
end

--=========================
-- // NEURON ACTIVATIONS
--=========================

-- Sigmoid(): Sigmoid activation function
-- @param x: Input value
-- @return number: Output between 0 and 1
function FunctionLib.Sigmoid(x: number): number
	if x >= 0 then
		return 1 / (1 + math.exp(-x))
	else
		return math.exp(x) / (1 + math.exp(x))
	end
end

-- SigmoidDerivative(): Sigmoid derivative
-- @param number: Output of the Sigmoid function
-- @return number: Derivative value
function FunctionLib.SigmoidDerivative(y: number): number
	return y * (1 - y)
end

-- ReLU(): ReLU activation function
-- @param x: Input value
-- @return number: math.max(0, x)
function FunctionLib.ReLU(x: number): number
	return math.max(0, x)
end

-- ReLUDerivative(): ReLU derivative
-- @param y: Output of the ReLU function
-- @return number: Derivative value
function FunctionLib.ReLUDerivative(y: number): number
	if y > 0 then
		return 1
	end

	return 0
end

-- LeakyReLU(): Leaky ReLU activation function
-- @param x: Input value
-- @return number: x if positive, alpha * x otherwise
function FunctionLib.LeakyReLU(x: number): number
	local alpha = 0.01

	if x > 0 then
		return x
	else
		return alpha * x
	end
end

-- LeakyReLUDerivative(): Leaky ReLU derivative
-- @param x: Input value
-- @return number: Derivative value
function FunctionLib.LeakyReLUDerivative(x: number): number
	local alpha = 0.01

	if x > 0 then
		return 1
	else
		return alpha
	end
end

-- TanhDerivative(): Tanh derivative
-- @param y: Output of Tanh function
-- @return number: Derivative value
function FunctionLib.TanhDerivative(y: number): number
	return 1 - y * y
end

return FunctionLib