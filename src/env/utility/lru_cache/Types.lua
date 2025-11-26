--!strict

export type lru_type<key, value> = {
	capacity: number,
	current_cache: { [key]: value },
	order: { key }
}

return nil