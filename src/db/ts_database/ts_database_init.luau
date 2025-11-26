--!strict

local ts_database = {}

--=========================
-- // SERVICES
--=========================

local HttpService = game:GetService("HttpService")

--=========================
-- // MODULES
--=========================

local Types = require("./Types")

local config = require("../../config")

--=========================
-- // DATA
--=========================

ts_database.api_key = ""

--=========================
-- // DATABASE OPERATIONS
--=========================

-- set_api_key(): Sets the API Key
-- @param api_key: The API key
-- @return boolean: If it was set
function ts_database.set_api_key(api_key: string): boolean
	local insert_url = config.strings.db_web_url .. "/set_api_key"

	local body = HttpService:JSONEncode({ apiKey = api_key })

	local success, response = pcall(function()
		return HttpService:PostAsync(insert_url, body, Enum.HttpContentType.ApplicationJson)
	end)

	if success then
		local response_data = HttpService:JSONDecode(response)

		ts_database.api_key = api_key

		return response_data.success
	else
		warn("set_api_key(): Failed to set API key: " .. response)
	end

	return false
end

-- insert(): Inserts data into the Database
-- @param data: The data to insert
-- @return ID: The ID of the inserted data or 0 if it failed
function ts_database.insert(data: Types.Data): number
	local insert_url = config.strings.db_web_url .. "/insert"

	local request_options = {
		Method = "POST",
		Url = insert_url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		},
		Body = HttpService:JSONEncode(data)
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		local response_data = HttpService:JSONDecode(response.Body)
		return response_data.id
	else
		warn("insert(): Failed to insert data: " .. (response and response.StatusMessage or response))
		return 0
	end
end

-- insert_temp(): Inserts temporary data into the Database
-- @param data: The data to insert
-- @return ID: The ID of the inserted data or 0 if it failed
function ts_database.insert_temp(data: Types.Data): number
	local insert_url = config.strings.db_web_url .. "/insert_temp"

	local request_options = {
		Method = "POST",
		Url = insert_url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		},
		Body = HttpService:JSONEncode(data)
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		local response_data = HttpService:JSONDecode(response.Body)
		return response_data.id
	else
		warn("insert_temp(): Failed to insert data: " .. (response and response.StatusMessage or response))
		return 0
	end
end

-- get(): Retrieves data by ID from the Database
-- @param id: The ID of the data to retrieve
-- @return any?: The requested data or nil if not found
function ts_database.get(id: number): any?
	local get_url = config.strings.db_web_url .. "/get/" .. tostring(id)

	local request_options = {
		Method = "GET",
		Url = get_url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		}
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		return HttpService:JSONDecode(response.Body)
	else
		warn("get(): Failed to fetch data for ID " .. tostring(id) .. ": " .. (response and response.StatusMessage or response))
		return nil
	end
end

-- filter(): Filters data in the Database based on criteria
-- @param criteria: The criteria to filter by
-- @return result: The filtered data
function ts_database.filter(criteria: Types.Data): { Types.Data }
	local filter_url = config.strings.db_web_url .. "/filter"

	local request_options = {
		Method = "POST",
		Url = filter_url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		},
		Body = HttpService:JSONEncode(criteria)
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		return HttpService:JSONDecode(response.Body)
	else
		warn("filter(): Failed to filter data: " .. (response and response.StatusMessage or response))
		return {}
	end
end

-- update(): Updates data in the Database
-- @param id: The ID of the data to update
-- @param data: The data to update with
-- @return success: Whether the update was successful
function ts_database.update(id: number, data: Types.Data): boolean
	local update_url = config.strings.db_web_url .. "/update/" .. tostring(id)

	local request_options = {
		Method = "PATCH",
		Url = update_url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		},
		Body = HttpService:JSONEncode(data)
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success then
		if response.StatusCode == 200 then
			return true
		else
			warn("update(): Failed to update data for ID " .. tostring(id) .. ": " .. response.StatusCode .. " - " .. response.StatusMessage)
			return false
		end
	else
		warn("update(): Error sending request: " .. response)
		return false
	end
end

-- delete(): Deletes data by ID from the Database
-- @param id: The ID of the data to delete
-- @return success: Whether the deletion was successful
function ts_database.delete(id: number): boolean
	local delete_url = config.strings.db_web_url .. "/delete/" .. tostring(id)

	local request_options = {
		Method = "DELETE",
		Url = delete_url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		}
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		return true
	else
		warn("delete(): Failed to delete data for ID " .. tostring(id) .. ": " .. (response and response.StatusMessage or response))
		return false
	end
end

-- cancel_temp_delete(): Cancels a temporary deletion operation for a given ID
-- @param id: The ID of the data whose temporary deletion should be canceled
-- @return success: Whether the cancellation was successful
function ts_database.cancel_temp_delete(id: number): boolean
	local cancel_url = config.strings.db_web_url .. "/cancel_temp/" .. tostring(id)

	local request_options = {
		Method = "PATCH",
		Url = cancel_url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		}
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		return true
	else
		warn("cancel_temp_delete(): Failed to cancel temporary deletion for ID " .. tostring(id) .. ": " .. (response and response.StatusMessage or response))
		return false
	end
end

-- get_all(): Retrieves all data from the Database
-- @return data: All the data in the database
function ts_database.get_all(): { Types.Data }
	local all_url = config.strings.db_web_url .. "/all"

	local request_options = {
		Method = "GET",
		Url = all_url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		}
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		return HttpService:JSONDecode(response.Body)
	else
		warn("get_all(): Failed to fetch all data: " .. (response and response.StatusMessage or response))
		return {}
	end
end

--=========================
-- // VERSION CONTROL
--=========================

-- create_empty_version(): Creates a new empty version
-- @param version_name: The name of the version to create
-- @return boolean: Whether the creation was successful
function ts_database.create_empty_version(version_name: string): boolean
	local url = config.strings.db_web_url .. "/versions/create_empty/" .. version_name

	local request_options = {
		Method = "POST",
		Url = url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		},
		Body = HttpService:JSONEncode({})
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		local data = HttpService:JSONDecode(response.Body)

		return data.success
	else
		warn("create_empty_version(): Failed to create version '" .. version_name .. "': " .. (response and response.StatusMessage or response))
		return false
	end
end

-- create_version(): Creates a new version from the current database
-- @param version_name: The name of the version to create
-- @param chunk_size: Optional chunk size
-- @return boolean: Whether the creation was successful
function ts_database.create_version(version_name: string, chunk_size: number?): boolean
	local url = config.strings.db_web_url .. "/versions/create/" .. version_name

	local body = {}
	if chunk_size then
		body.chunkSize = chunk_size
	end

	local request_options = {
		Method = "POST",
		Url = url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		},
		Body = HttpService:JSONEncode(body)
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		local data = HttpService:JSONDecode(response.Body)
		return data.success
	else
		warn("create_version(): Failed to create version '" .. version_name .. "': " .. (response and response.StatusMessage or response))
		return false
	end
end

-- load_version(): Loads a specific version into the database (restores)
-- @param version_name: The name of the version to load
-- @return boolean: Whether the load was successful
function ts_database.load_version(version_name: string): boolean
	local url = config.strings.db_web_url .. "/versions/load/" .. version_name

	local request_options = {
		Method = "POST",
		Url = url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		},
		Body = HttpService:JSONEncode({})
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		local data = HttpService:JSONDecode(response.Body)
		return data.success
	else
		warn("load_version(): Failed to load version '" .. version_name .. "': " .. (response and response.StatusMessage or response))
		return false
	end
end

-- delete_version(): Deletes a version by name
-- @param version_name: The name of the version to delete
-- @return boolean: Whether the deletion was successful
function ts_database.delete_version(version_name: string): boolean
	local url = config.strings.db_web_url .. "/versions/delete/" .. version_name

	local request_options = {
		Method = "DELETE",
		Url = url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		}
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		return true
	else
		warn("delete_version(): Failed to delete version '" .. version_name .. "': " .. (response and response.StatusMessage or response))
		return false
	end
end

-- list_versions(): Lists all available versions
-- @return { string }: Array of version names
function ts_database.list_versions(): { string }
	local url = config.strings.db_web_url .. "/versions/all"

	local request_options = {
		Method = "GET",
		Url = url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		}
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response and response.StatusCode == 200 then
		local data = HttpService:JSONDecode(response.Body)

		return data.versions
	else
		warn("list_versions(): Failed to list versions: " .. (response and response.StatusMessage or tostring(response)))
		return {}
	end
end

-- get_metadata(): Get metadata for a specific version (Snapshot of when created, doesn't change). (Default doesnt have any metadata)
-- @param name: The name of the version to get metadata for
-- @return { any }: The metadata
function ts_database.get_metadata(name: string): { any }
	local url = config.strings.db_web_url .. "/versions/metadata/" .. name

	local request_options = {
		Method = "GET",
		Url = url,
		Headers = {
			["Content-Type"] = "application/json",
			["api-key"] = ts_database.api_key
		}
	}

	local success, response = pcall(function()
		return HttpService:RequestAsync(request_options)
	end)

	if success and response.StatusCode == 200 then
		local data = HttpService:JSONDecode(response.Body)
		return data.metadata or {}
	else
		warn("get_metadata(): Failed to get metadata: " .. (response and response.StatusMessage or response))
		return {}
	end
end

return ts_database