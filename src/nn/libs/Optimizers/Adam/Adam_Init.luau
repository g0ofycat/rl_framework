--!strict

--=========================
-- // MODULES
--=========================

local Config = require("./Config")

--=========================
-- // OPTIMIZER API
--=========================

return
    -- Adam(): Adam optimizer
    -- @param learning_rate: number
    -- @param gradients: number
    -- @param time_step: number
    -- @param state: { m: number, v: number }
    -- @return number: Updated parameter value
    function(
        learning_rate: number,
        gradients: number,
        time_step: number,
        state: { m: number, v: number }
    ): number
        local m = state.m
        local v = state.v

        m = Config.First_Moment_Decay * m + (1 - Config.First_Moment_Decay) * gradients
        v = Config.Second_Moment_Decay * v + (1 - Config.Second_Moment_Decay) * (gradients * gradients)

        local m_hat = m / (1 - Config.First_Moment_Decay ^ time_step)
        local v_hat = v / (1 - Config.Second_Moment_Decay ^ time_step)

        local update = learning_rate * (m_hat / (math.sqrt(v_hat) + Config.Epsilon))

        state.m = m
        state.v = v

        return update
    end