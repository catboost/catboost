/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <memory>
#include <string>
#include "shareddata.h"

namespace kiwi
{

class Variable
{

public:
    class Context
    {
    public:
        Context() = default;
        virtual ~Context() {} // LCOV_EXCL_LINE
    };

    Variable(Context *context = 0) : m_data(new VariableData("", context)) {}

    Variable(std::string name, Context *context = 0) : m_data(new VariableData(std::move(name), context)) {}

    Variable(const char *name, Context *context = 0) : m_data(new VariableData(name, context)) {}

    Variable(const Variable&) = default;

    Variable(Variable&&) noexcept = default;

    ~Variable() = default;

    const std::string &name() const
    {
        return m_data->m_name;
    }

    void setName(const char *name)
    {
        m_data->m_name = name;
    }

    void setName(const std::string &name)
    {
        m_data->m_name = name;
    }

    Context *context() const
    {
        return m_data->m_context.get();
    }

    void setContext(Context *context)
    {
        m_data->m_context.reset(context);
    }

    double value() const
    {
        return m_data->m_value;
    }

    void setValue(double value)
    {
        m_data->m_value = value;
    }

    // operator== is used for symbolics
    bool equals(const Variable &other)
    {
        return m_data == other.m_data;
    }

    Variable& operator=(const Variable&) = default;

    Variable& operator=(Variable&&) noexcept = default;

private:
    class VariableData : public SharedData
    {

    public:
        VariableData(std::string name, Context *context) : SharedData(),
                                                                  m_name(std::move(name)),
                                                                  m_context(context),
                                                                  m_value(0.0) {}

        VariableData(const char *name, Context *context) : SharedData(),
                                                           m_name(name),
                                                           m_context(context),
                                                           m_value(0.0) {}

        ~VariableData() = default;

        std::string m_name;
        std::unique_ptr<Context> m_context;
        double m_value;

    private:
        VariableData(const VariableData &other);

        VariableData &operator=(const VariableData &other);
    };

    SharedDataPtr<VariableData> m_data;

    friend bool operator<(const Variable &lhs, const Variable &rhs)
    {
        return lhs.m_data < rhs.m_data;
    }
};

} // namespace kiwi
