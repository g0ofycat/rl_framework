--!strict

local lru_cache = {}

lru_cache.__index = lru_cache

--=========================
-- // SERVICES
--=========================

local HttpService = game:GetService("HttpService")

--=========================
-- // TYPES
--=========================

local Types = require("./Types")

export type lru_cache_constructor_type<key, value> = typeof(setmetatable({} :: Types.lru_type<key, value>, lru_cache))

--=========================
-- // CONSTRUCTOR
--=========================

-- new(): New LRU Cache Object
-- @param capacity: Cache capacity
-- @return lru_cache_constructor_type<key, value>
function lru_cache.new<key, value>(capacity: number): lru_cache_constructor_type<key, value>
	local self = {}

	self.capacity = capacity

	self.current_cache = {}

	self.order = {}

	return setmetatable(self, lru_cache)
end

--=========================
-- // PUBLIC API
--=========================

-- get(): Get data from the cache
-- @param key: The key to get
-- @return value?: The value of the key
function lru_cache:get<key, value>(key: key): value?
	local value = self.current_cache[key]

	if value == nil then
		return nil
	end

	for i = 1, #self.order do
		if self.order[i] == key then
			table.remove(self.order, i)
			break
		end
	end

	table.insert(self.order, key)

	return value
end

-- put(): Put data into the cache
-- @param key: The key to put
-- @param value: The value to put
function lru_cache:put<key, value>(key: key, value: value): ()
	if self.current_cache[key] ~= nil then
		for i = 1, #self.order do
			if self.order[i] == key then
				table.remove(self.order, i)
				break
			end
		end
	else
		if #self.order >= self.capacity then
			local oldest = table.remove(self.order, 1) :: key
			self.current_cache[oldest] = nil
		end
	end

	table.insert(self.order, key)

	self.current_cache[key] = value
end

-- size(): Get the size of the cache
-- @return number: Size of the cache
function lru_cache:size(): number
	return #self.order
end

-- put_array(): Insert an array into the cache
-- @param array: The array to put
function lru_cache:put_array(array: { any }): ()
	local key = HttpService:JSONEncode(array)

	if self.current_cache[key] ~= nil then
		for i = 1, #self.order do
			if self.order[i] == key then
				table.remove(self.order, i)
				break
			end
		end
	else
		if #self.order >= self.capacity then
			local oldest = table.remove(self.order, 1)
			self.current_cache[oldest] = nil
		end
	end

	table.insert(self.order, key)

	self.current_cache[key] = array
end

-- reset(): Reset the cache
function lru_cache:reset(): ()
	self.current_cache = {}
	self.order = {}
end

-- export_data(): Export the cache data
-- @return Types.lru_type<key, value>
function lru_cache:export_data<key, value>(): Types.lru_type<key, value>
	return {
		capacity = self.capacity,

		current_cache = self.current_cache,

		order = self.order
	}
end

return lru_cache