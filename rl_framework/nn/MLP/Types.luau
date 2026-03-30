--!strict

export type OptimizerState = { m: number, v: number }

export type NN_Weights = { [number]: { [number]: { [number]: number } } }
export type NN_Biases = { [number]: { [number]: number } }

export type XavierUniformType = (weights: NN_Weights, fromSize: number, toSize: number, layer: number, random_generator: Random) -> ()

export type HeUniformType = (weights: NN_Weights, fromSize: number, toSize: number, layer: number, random_generator: Random) -> ()

export type ConfigType = {
	activation_function: {
		Activation: (number) -> number,
		Derivative: (number) -> number
	},

	loss_function: ({ number }, { number }) -> number,

	optimizer_function: any,

	classification_function: ({ number }) -> { number },

	weight_init_function: XavierUniformType | HeUniformType,

	[any]: any
}

export type OptionalConfigType = {
	activation_function: {
		Activation: ((number) -> number)?,
		Derivative: ((number) -> number)?
	}?,

	loss_function: (({ number }, { number }) -> number)?,

	optimizer_function: any?,

	classification_function: (({ number }) -> { number })?,

	weight_init_function: (XavierUniformType | HeUniformType)?,

	[any]: any
}

export type MLP_type = {
	inputNodes: number,
	hiddenLayers: { number },
	outputNodes: number,

	dropout_rate: number,

	random_generator: Random,

	optimizer_function: any,

	weights: NN_Weights,
	biases: NN_Biases,

	optimizerStateWeights: { [number]: { [number]: { [number]: OptimizerState } } },
	optimizerStateBiases: { [number]: { [number]: OptimizerState } },

	dropoutMasks: { [number]: { [number]: number } },

	config: ConfigType,

	layerSizes: { number }
}

export type NN_Data = {
	inputNodes: number,
	hiddenLayers: { number },
	layerSizes: { number },
	outputNodes: number,

	weights: NN_Weights,
	biases: NN_Biases,

	min: number?,
	scale: number?
}

return nil