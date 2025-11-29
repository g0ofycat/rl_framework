--!strict

export type AdamState = { m: number, v: number }

export type MLP_type = {
	inputNodes: number,
	hiddenLayers: { number },
	outputNodes: number,

	dropout_rate: number,

	timeStep: number,

	random_generator: Random,

	weights: { [number]: { [number]: { [number]: number } } },
	biases: { [number]: { [number]: number } },

	adamStateWeights: { [number]: { [number]: { [number]: AdamState } } },
	adamStateBiases: { [number]: { [number]: AdamState } },

	dropoutMasks: { [number]: { [number]: number } }
}

export type NN_Data = {
	weights: { [number]: { [number]: { [number]: number } } },

	biases: { [number]: { [number]: number } },

	hiddenLayers: { number }
}

return nil