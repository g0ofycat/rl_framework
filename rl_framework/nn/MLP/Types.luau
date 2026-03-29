--!strict

export type OptimizerState = { m: number, v: number }

export type XavierUniformType = (weights: { [number]: { [number]: { [number]: number } } }, fromSize: number, toSize: number, layer: number, random_generator: Random) -> ()

export type HeUniformType = (weights: { [number]: { [number]: { [number]: number } } }, fromSize: number, toSize: number, layer: number, random_generator: Random) -> ()

export type ConfigType = {
	activation_function: {
		Activation: (number) -> number,
		Derivative: (number) -> number
	},

	loss_function: ({ number }, { number }) -> number,

	optimizer_function: any,

	classification_function: ({ number }) -> { number },

	weight_init_function: XavierUniformType | HeUniformType
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