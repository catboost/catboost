local function off(i)
    local ss = ''
    local j = 0

    while j < i do
        ss = ss .. '    '
        j = j + 1
    end

    return ss
end

local function fmtkey(key)
    if type(key) == 'string' and key:match('^[_%a][_%w]*$') then
        return key
    end

    return '[' .. string.format('%q', key) .. ']'
end

local function pp(v, i)
    if type(v) == "string" then
        return string.format('%q', v)
    end

    if type(v) == "number" then
        return tostring(v)
    end

    if type(v) == "nil" then
        return 'nil'
    end

    if type(v) == "boolean" then
        return tostring(v)
    end

    if type(v) == "table" then
        local ret = "{\n"
        local curoff = 1

        for x, y in pairs(v) do
            if type(x) == 'number' and x == curoff then
                ret = ret .. off(i + 1) .. pp(y, i + 1) .. ';\n'
                curoff = curoff + 1
            else
                ret = ret .. off(i + 1) .. fmtkey(x) .. " = " .. pp(y, i + 1) .. ';\n'
            end
        end

        return ret .. off(i) .. "}"
    end

    if v == nil then
        return 'nil'
    end

    return ""
end

local function prettify(v)
    return pp(v, 0)
end
