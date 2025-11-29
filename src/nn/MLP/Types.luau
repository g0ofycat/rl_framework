--!strict

export type OptimizerState = { m: number, v: number }

export type ConfigType = {
	activation_function: {
		Activation: (number) -> number,
		Derivative: (number) -> number
	},

	loss_function: ({ number }, { number }) -> number,

	optimizer_function: any,

	classification_function: ({ number }) -> { number }
}

export type MLP_type = {
	inputNodes: number,
	hiddenLayers: { number },
	outputNodes: number,

	dropout_rate: number,

	random_generator: Random,

	optimizer_function: any,

	weights: { [number]: { [number]: { [number]: number } } },
	biases: { [number]: { [number]: number } },

	optimizerStateWeights: { [number]: { [number]: { [number]: OptimizerState } } },
	optimizerStateBiases: { [number]: { [number]: OptimizerState } },

	dropoutMasks: { [number]: { [number]: number } },

	config: ConfigType
}

export type NN_Data = {
	weights: { [number]: { [number]: { [number]: number } } },

	biases: { [number]: { [number]: number } },

	hiddenLayers: { number }
}

return nil