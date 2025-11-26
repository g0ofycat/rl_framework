--!strict

export type AdamState = { m: number, v: number }

export type FFN_type = {
	inputNodes: number,
	hiddenNodes: number,
	outputNodes: number,

	dropout_rate: number,

	timeStep: number,

	weightsIH: { [number]: { [number]: number } },
	weightsHO: { [number]: { [number]: number } },
	biasH: { [number]: number },
	biasO: { [number]: number },

	adamStateIH: { [number]: { [number]: AdamState } },
	adamStateHO: { [number]: { [number]: AdamState } },
	adamStateBiasH: { [number]: AdamState },
	adamStateBiasO: { [number]: AdamState },

	dropoutMask: { [number]: number }
}

export type NN_Data = {
	weightsIH: { [number]: { [number]: number } },
	weightsHO: { [number]: { [number]: number } },

	biasH: { [number]: number },
	biasO: { [number]: number }
}

return nil