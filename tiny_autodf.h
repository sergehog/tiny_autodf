/*
 * This file is part of the Tiny-DF distribution (https://github.com/sergehog/tiny_df)
 * Copyright (c) 2020-2021 Sergey Smirnov / Seregium Oy.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

//
// Simple automatic differentiation C++/header-only library
// Supposed to be very easy in use, and be able to replace/represent classical scalars types in basic formulas/equations
// Supports partial derivatives (i.e. functions of many variables)
//

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <memory>
#include <unordered_map>

#ifndef TINY_AUTODF_H_
#define TINY_AUTODF_H_

namespace tiny_autodf
{

template <typename ScalarType = float>
class AutoDf
{
  public:
    /// Each AutoDf despite its AutoType has ID of this type
    using IdType = size_t;

    /// Type for storing result of AutoDf formula evaluation
    struct Evaluation
    {
        ScalarType value;
        std::unordered_map<IdType, ScalarType> derivatives;
    };

    /// List of all implemented AutoDf variants and operations
    enum class AutoType
    {
        kConstType,     /// just a const, cannot be changed
        kVariableType,  /// input variable, may be assigned/changed
        kSumType,       /// result of sum of two input AutoDfs
        kSubtractType,  /// result of subtract of two input AutoDfs
        kMultType,      /// result of multiplication of two input AutoDfs
        kDivType,       /// result of divition of two input AutoDfs
        kAbsType,       /// absolute value of one input AutoDf
        kMaxType,       /// maximum value among two input AutoDfs
        kMinType,       /// minumum value among two input AutoDfs
        kSinType,       /// sin() value of one input AutoDf
        kCosType        /// cos() value of one input AutoDf
    };

  private:
    /// If bulk creation of Variables is enabled, it keeps corresponding value
    static std::atomic<bool> create_variables_;

    ///  Latest assigned ID value
    static std::atomic<IdType> id_increment_;

    struct CallGraphNode
    {
        IdType ID{};
        AutoType type{};
        size_t count{};
        std::shared_ptr<CallGraphNode> left{};
        std::shared_ptr<CallGraphNode> right{};
        std::shared_ptr<ScalarType> value{};
        std::unordered_map<size_t, std::shared_ptr<ScalarType>> variables{};

        CallGraphNode(const AutoType Type, const ScalarType& value_ = static_cast<ScalarType>(0.))
            : ID(id_increment_++), type(Type)
        {
            value = std::make_shared<ScalarType>(value_);
            count = 1;
        }

        /// Evaluation of call-graph,
        /// @returns evaluated value & values of derivatives
        Evaluation eval()
        {
            Evaluation eval_out{};
            if (type == AutoType::kConstType)
            {
                eval_out.value = *value;
                return eval_out;
            }
            else if (type == AutoType::kVariableType)
            {
                eval_out.value = *value;
                eval_out.derivatives[ID] = static_cast<ScalarType>(1.);
                return eval_out;
            }

            const auto eval1 = left->eval();
            const ScalarType v1 = eval1.value;

            if (type == AutoType::kAbsType)
            {
                *value = eval_out.value = std::abs(v1);
                const bool sign_changed = v1 < ScalarType(0.);
                for (auto vi = variables.begin(); vi != variables.end(); vi++)
                {
                    const size_t id = vi->first;
                    const float d1 = eval1.derivatives.at(id);
                    eval_out.derivatives[id] = sign_changed ? -d1 : d1;
                }
                return eval_out;
            }
            else if (type == AutoType::kSinType)
            {
                *value = eval_out.value = std::sin(v1);

                for (auto vi = variables.begin(); vi != variables.end(); vi++)
                {
                    const size_t id = vi->first;
                    const float d1 = eval1.derivatives.at(id);
                    eval_out.derivatives[id] = std::cos(eval1.value) * d1;
                }
                return eval_out;
            }
            else if (type == AutoType::kCosType)
            {
                *value = eval_out.value = std::cos(v1);

                for (auto vi = variables.begin(); vi != variables.end(); vi++)
                {
                    const size_t id = vi->first;
                    const float d1 = eval1.derivatives.at(id);
                    eval_out.derivatives[id] = -std::sin(eval1.value) * d1;
                }
                return eval_out;
            }

            const auto eval2 = right->eval();
            const ScalarType v2 = eval2.value;

            if (type == AutoType::kSumType)
            {
                *value = eval_out.value = v1 + v2;

                for (auto vi = variables.begin(); vi != variables.end(); vi++)
                {
                    const size_t id = vi->first;
                    eval_out.derivatives[id] = static_cast<ScalarType>(0.);
                    const auto d1 = eval1.derivatives.find(id);
                    if (d1 != eval1.derivatives.end())
                    {
                        eval_out.derivatives[id] += d1->second;
                    }
                    const auto d2 = eval2.derivatives.find(id);
                    if (d2 != eval2.derivatives.end())
                    {
                        eval_out.derivatives[id] += d2->second;
                    }
                }
            }
            else if (type == AutoType::kSubtractType)
            {
                *value = eval_out.value = v1 - v2;
                for (auto vi = variables.begin(); vi != variables.end(); vi++)
                {
                    const size_t id = vi->first;
                    eval_out.derivatives[id] = static_cast<ScalarType>(0.);
                    const auto d1 = eval1.derivatives.find(id);
                    if (d1 != eval1.derivatives.end())
                    {
                        eval_out.derivatives[id] += d1->second;
                    }
                    const auto d2 = eval2.derivatives.find(id);
                    if (d2 != eval2.derivatives.end())
                    {
                        eval_out.derivatives[id] -= d2->second;
                    }
                }
            }
            else if (type == AutoType::kMultType)
            {
                *value = eval_out.value = v1 * v2;

                for (auto vi = variables.begin(); vi != variables.end(); vi++)
                {
                    const size_t id = vi->first;
                    eval_out.derivatives[id] = static_cast<ScalarType>(0.);
                    const auto d1 = eval1.derivatives.find(id);
                    if (d1 != eval1.derivatives.end())
                    {
                        const ScalarType g1 = d1->second;
                        eval_out.derivatives[id] += v2 * g1;
                    }
                    const auto d2 = eval2.derivatives.find(id);
                    if (d2 != eval2.derivatives.end())
                    {
                        const ScalarType g2 = d2->second;
                        eval_out.derivatives[id] += v1 * g2;
                    }
                }
            }
            else if (type == AutoType::kDivType)
            {
                *value = eval_out.value = v1 / v2;

                for (auto vi = variables.begin(); vi != variables.end(); vi++)
                {
                    const size_t id = vi->first;
                    const auto g1 = eval1.derivatives.find(id);
                    const auto g2 = eval2.derivatives.find(id);
                    if (g1 != eval1.derivatives.end())
                    {
                        const ScalarType d1 = g1->second;

                        eval_out.derivatives[id] += d1 * v2 / (v2 * v2);
                    }
                    if (g2 != eval2.derivatives.end())
                    {
                        const ScalarType d2 = g2->second;
                        eval_out.derivatives[id] -= d2 * v1 / (v2 * v2);
                    }
                }
            }
            else if (type == AutoType::kMaxType || type == AutoType::kMinType)
            {
                const bool left_selected = type == AutoType::kMaxType ? v1 >= v2 : v1 <= v2;
                *value = eval_out.value = left_selected ? v1 : v2;

                for (auto vi = variables.begin(); vi != variables.end(); vi++)
                {
                    const size_t id = vi->first;
                    const auto d1 = eval1.derivatives.find(id);
                    const auto d2 = eval2.derivatives.find(id);
                    if (d1 != eval1.derivatives.end() && left_selected)
                    {
                        eval_out.derivatives[id] = d1->second;
                    }
                    else
                    {
                        eval_out.derivatives[id] = ScalarType(0.);
                    }
                    if (d2 != eval1.derivatives.end() && !left_selected)
                    {
                        if (d2 != eval1.derivatives.end())
                        {
                            eval_out.derivatives[id] = d2->second;
                        }
                        else
                        {
                            eval_out.derivatives[id] = ScalarType(0.);
                        }
                    }
                }
            }

            return eval_out;
        };
    };

    /// Graph Node belonging to current variable / AutoDf instance
    std::shared_ptr<CallGraphNode> node_{};

    /// In order not to multiply zero-valued ConstType nodes, we maintain single such node and re-use
    /// Reduces memory-consumption as well unpredictable id_increment grow
    static const std::shared_ptr<CallGraphNode> zero_;

    /// Creates AutoDf of specified type, and takes their ownership if needed
    AutoDf(const AutoType type,
           const std::shared_ptr<CallGraphNode>& left,
           const std::shared_ptr<CallGraphNode>& right,
           const ScalarType& value = static_cast<ScalarType>(0.))
    {
        node_ = std::make_shared<CallGraphNode>(type, value);
        node_->left = left;
        node_->right = right;
        node_->count = 0U;
        if (left)
        {
            node_->count += left->count;
            node_->variables.insert(left->variables.begin(), left->variables.end());
        }
        if (right)
        {
            node_->count += right->count;
            node_->variables.insert(right->variables.begin(), right->variables.end());
        }
    }

    AutoDf(const std::shared_ptr<CallGraphNode>& node_in) : node_(node_in) {}

  public:
    /// Read-only ID value, could be used to distinguish partial derivative of this variable
    IdType ID() const { return node_->ID; }

    size_t count() const { return node_->count; }

    /// Explicitly set what type of AutoDf will be created by-default
    static void StartConstants(const bool need_constant = true) { create_variables_ = !(need_constant); }

    static void StartVariables(const bool need_variable = true) { create_variables_ = need_variable; }

    /// Creates kVariableType or kConstType (depending on default_type_) with zero value
    AutoDf()
    {
        if (create_variables_)
        {
            node_ = std::make_shared<CallGraphNode>(AutoType::kVariableType);
            node_->variables[node_->ID] = node_->value;
        }
        else
        {
            node_ = zero_;
        }
    }

    /// Creates kVariableType or kConstType (depending on default_type_) with specified value
    AutoDf(const ScalarType& scalar)
    {
        if (!create_variables_ && scalar == static_cast<ScalarType>(0))
        {
            // avoid creation trivial const-type node
            node_ = zero_;
        }
        else
        {
            const auto default_type = create_variables_ ? AutoType::kVariableType : AutoType::kConstType;
            node_ = std::make_shared<CallGraphNode>(default_type, scalar);
            if (default_type == AutoType::kVariableType)
            {
                node_->variables[node_->ID] = node_->value;
            }
        }
    }

    /// Copy-Constructor, re-uses computation graph node
    AutoDf(const AutoDf<ScalarType>& other) { node_ = other.node_; }

    /// Additional way to create AutoDf of specified type
    AutoDf(const ScalarType& value, const bool is_const)
    {
        const auto type = (is_const ? AutoType::kConstType : AutoType::kVariableType);
        node_ = std::make_shared<CallGraphNode>(type, value);
        if (node_->type == AutoType::kVariableType)
        {
            node_->variables[node_->ID] = node_->value;
        }
    }

    AutoDf& operator=(const ScalarType& scalar)
    {
        if (node_ == zero_ && scalar != ScalarType(0))
        {
            const auto default_type = create_variables_ ? AutoType::kVariableType : AutoType::kConstType;
            node_ = std::make_shared<CallGraphNode>(default_type, scalar);
            if (default_type == AutoType::kVariableType)
            {
                node_->variables[node_->ID] = node_->value;
            }
        }
        else if (node_->type == AutoType::kConstType || node_->type == AutoType::kVariableType)
        {
            *node_->value = scalar;
        }
        return *this;
    }

    AutoDf<ScalarType>& operator=(const AutoDf<ScalarType>& other)
    {
        node_ = other.node_;
        return *this;
    }

    /// List of input "mutable" values, contributing to this AutoDf value
    std::unordered_map<size_t, std::shared_ptr<ScalarType>>& variables() const { return node_->variables; }

    Evaluation eval() const { return node_->eval(); }

    /// @returns latest calculated value or sets new one (but only for variables)
    /// One may change underlying value directly
    ScalarType& value() const
    {
        if (node_->type == AutoType::kVariableType)
        {
            return *node_->value;
        }
        else
        {
            // prohibit exposing real value if this node is not of the variable type
            static ScalarType fake_value = ScalarType(0);
            fake_value = *node_->value;
            return fake_value;
        }
    }

    explicit operator ScalarType() const { return *node_->value; }

    AutoDf<ScalarType>& operator+=(const ScalarType value)
    {
        // avoid creation graph nodes, when not really needed
        // (adding with const zero wont change anything)
        if (value == static_cast<ScalarType>(0))
        {
            return *this;
        }
        else if (node_ == zero_)
        {
            node_ = std::make_shared<CallGraphNode>(AutoType::kConstType, value);
            return *this;
        }
        else if (node_->type == AutoType::kConstType)
        {
            const ScalarType old_node_value = *node_->value;
            node_ = std::make_shared<CallGraphNode>(AutoType::kConstType, old_node_value + value);
            return *this;
        }

        AutoDf<ScalarType> other(value, true);
        return InPlaceOperator(AutoType::kSumType, other.node_, *node_->value + *other.node_->value);
    }

    AutoDf<ScalarType>& operator+=(const AutoDf<ScalarType>& other)
    {
        // avoid creation graph nodes, when not really needed
        if (other.node_->type == AutoType::kConstType && *other.node_->value == static_cast<ScalarType>(0))
        {
            return *this;
        }
        else if (node_->type == AutoType::kConstType && other.node_->type == AutoType::kConstType)
        {
            const ScalarType node_value = *node_->value;
            node_ = std::make_shared<CallGraphNode>(AutoType::kConstType, node_value + *other.node_->value);
            return *this;
        }

        return InPlaceOperator(AutoType::kSumType, other.node_, *node_->value + *other.node_->value);
    }

    AutoDf<ScalarType>& operator-=(const ScalarType value)
    {
        // avoid creating graph nodes, when not really needed
        if (value == static_cast<ScalarType>(0))
        {
            return *this;
        }
        else if (node_->type == AutoType::kConstType)
        {
            const ScalarType node_value = *node_->value;
            node_ = std::make_shared<CallGraphNode>(AutoType::kConstType, node_value - value);
            return *this;
        }

        AutoDf<ScalarType> other(value, true);
        return InPlaceOperator(AutoType::kSubtractType, other.node_, *node_->value - *other.node_->value);
    }

    AutoDf<ScalarType>& operator-=(const AutoDf<ScalarType>& other)
    {
        if (other.node_->type == AutoType::kConstType && *other.node_->value == static_cast<ScalarType>(0))
        {
            return *this;
        }
        else if (node_->type == AutoType::kConstType && other.node_->type == AutoType::kConstType)
        {
            const ScalarType node_value = *node_->value;
            node_ = std::make_shared<CallGraphNode>(AutoType::kConstType, node_value - *other.node_->value);
            return *this;
        }

        return InPlaceOperator(AutoType::kSubtractType, other.node_, *node_->value - *other.node_->value);
    }

    static AutoDf<ScalarType> abs(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoType::kAbsType, other.node_, nullptr, std::abs(*other.node_->value));
    }

    static AutoDf<ScalarType> min(const AutoDf<ScalarType>& left, const AutoDf<ScalarType>& right)
    {
        return AutoDf<ScalarType>(
            AutoType::kMinType, left.node_, right.node_, std::min(*left.node_->value, *right.node_->value));
    }

    static AutoDf<ScalarType> max(const AutoDf<ScalarType>& left, const AutoDf<ScalarType>& right)
    {
        return AutoDf<ScalarType>(
            AutoType::kMaxType, left.node_, right.node_, std::max(*left.node_->value, *right.node_->value));
    }

    static AutoDf<ScalarType> sin(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoType::kSinType, other.node_, nullptr, std::sin(*other.node_->value));
    }

    static AutoDf<ScalarType> cos(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoType::kCosType, other.node_, nullptr, std::cos(*other.node_->value));
    }

#define AUTODF_DEFINE_OPERATOR(op)                                              \
    template <typename T>                                                       \
    friend AutoDf<T> operator op(const AutoDf<T>& other, const T scalar_value); \
    template <typename T>                                                       \
    friend AutoDf<T> operator op(const T scalar_value, const AutoDf<T>& other); \
    template <typename T>                                                       \
    friend AutoDf<T> operator op(const AutoDf<T>& left, const AutoDf<T>& right);

    AUTODF_DEFINE_OPERATOR(+);
    AUTODF_DEFINE_OPERATOR(-);
    AUTODF_DEFINE_OPERATOR(*);
    AUTODF_DEFINE_OPERATOR(/);
#undef AUTODF_DEFINE_OPERATOR

    template <typename T>
    friend AutoDf<T> operator-(AutoDf<T> const& other);

  private:
    AutoDf<ScalarType>& InPlaceOperator(const AutoType& type,
                                        const std::shared_ptr<CallGraphNode>& right_node,
                                        const ScalarType new_value)
    {
        auto result = std::make_shared<CallGraphNode>(type, new_value);
        result->left = node_;
        result->right = right_node;
        result->count = node_->count + right_node->count;
        result->variables.insert(node_->variables.begin(), node_->variables.end());
        result->variables.insert(right_node->variables.begin(), right_node->variables.end());
        node_.swap(result);
        return *this;
    }

    static AutoDf<ScalarType> make_sum(const std::shared_ptr<CallGraphNode>& left,
                                       const std::shared_ptr<CallGraphNode>& right)
    {
        if (left->type == AutoType::kConstType && *left->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(right);
        }
        if (right->type == AutoType::kConstType && *right->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(left);
        }

        const ScalarType value = *left->value + *right->value;
        return AutoDf<ScalarType>(AutoType::kSumType, left, right, value);
    }

    static AutoDf<ScalarType> make_sub(const std::shared_ptr<CallGraphNode>& left,
                                       const std::shared_ptr<CallGraphNode>& right)
    {
        if (right->type == AutoType::kConstType && *right->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(left);
        }

        const ScalarType value = *left->value - *right->value;
        return AutoDf<ScalarType>(AutoType::kSubtractType, left, right, value);
    }

    static AutoDf<ScalarType> make_mult(const std::shared_ptr<CallGraphNode>& left,
                                        const std::shared_ptr<CallGraphNode>& right)
    {
        if (left->type == AutoType::kConstType && *left->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(zero_);
        }
        if (right->type == AutoType::kConstType && *right->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(zero_);
        }
        if (left->type == AutoType::kConstType && *left->value == static_cast<ScalarType>(1))
        {
            return AutoDf<ScalarType>(right);
        }
        if (right->type == AutoType::kConstType && *right->value == static_cast<ScalarType>(1))
        {
            return AutoDf<ScalarType>(left);
        }

        const ScalarType value = *left->value * *right->value;
        return AutoDf<ScalarType>(AutoType::kMultType, left, right, value);
    }

    static AutoDf<ScalarType> make_div(const std::shared_ptr<CallGraphNode>& left,
                                       const std::shared_ptr<CallGraphNode>& right)
    {
        if (left->type == AutoType::kConstType && *left->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(zero_);
        }
        if (right->type == AutoType::kConstType && *right->value == static_cast<ScalarType>(1))
        {
            return AutoDf<ScalarType>(left);
        }

        const ScalarType value = *left->value / *right->value;
        return AutoDf<ScalarType>(AutoType::kDivType, left, right, value);
    }
};

template <typename ScalarType>
AutoDf<ScalarType> operator-(AutoDf<ScalarType> const& other)
{
    if (other.node_->type == AutoDf<ScalarType>::AutoType::kConstType)
    {
        return AutoDf<ScalarType>(-other.value(), true);
    }
    else
    {
        return AutoDf<ScalarType>::make_sub(AutoDf<ScalarType>::zero_, other.node_);
    }
}

/// MACRO to helps define operators of 2 arguments
#define AUTODF_DEFINE_OPERATOR(OP, func)                                                            \
    template <typename ScalarType>                                                                  \
    AutoDf<ScalarType> operator OP(const AutoDf<ScalarType>& other, const ScalarType scalar_value)  \
    {                                                                                               \
        AutoDf<ScalarType> scalar(scalar_value, true);                                              \
        return AutoDf<ScalarType>::func(other.node_, scalar.node_);                                 \
    }                                                                                               \
    template <typename ScalarType>                                                                  \
    AutoDf<ScalarType> operator OP(const ScalarType scalar_value, const AutoDf<ScalarType>& other)  \
    {                                                                                               \
        AutoDf<ScalarType> scalar(scalar_value, true);                                              \
        return AutoDf<ScalarType>::func(scalar.node_, other.node_);                                 \
    }                                                                                               \
    template <typename ScalarType>                                                                  \
    AutoDf<ScalarType> operator OP(const AutoDf<ScalarType>& left, const AutoDf<ScalarType>& right) \
    {                                                                                               \
        return AutoDf<ScalarType>::func(left.node_, right.node_);                                   \
    }

/// Definition of math operators
AUTODF_DEFINE_OPERATOR(+, make_sum);
AUTODF_DEFINE_OPERATOR(-, make_sub);
AUTODF_DEFINE_OPERATOR(*, make_mult);
AUTODF_DEFINE_OPERATOR(/, make_div);
#undef AUTODF_DEFINE_OPERATOR

/// MACRO to helps define functions of 2 arguments
#define AUTODF_DEFINE_FUNCTION(func)                                                         \
    template <typename ScalarType>                                                           \
    AutoDf<ScalarType> func(const AutoDf<ScalarType>& other, const ScalarType scalar_value)  \
    {                                                                                        \
        AutoDf<ScalarType> scalar(scalar_value, true);                                       \
        return AutoDf<ScalarType>::func(other, scalar);                                      \
    }                                                                                        \
    template <typename ScalarType>                                                           \
    AutoDf<ScalarType> func(const ScalarType scalar_value, const AutoDf<ScalarType>& other)  \
    {                                                                                        \
        AutoDf<ScalarType> scalar(scalar_value, true);                                       \
        return AutoDf<ScalarType>::func(scalar, other);                                      \
    }                                                                                        \
    template <typename ScalarType>                                                           \
    AutoDf<ScalarType> func(const AutoDf<ScalarType>& left, const AutoDf<ScalarType>& right) \
    {                                                                                        \
        return AutoDf<ScalarType>::func(left, right);                                        \
    }

/// Definition of some functions
AUTODF_DEFINE_FUNCTION(min);
AUTODF_DEFINE_FUNCTION(max);
#undef AUTODF_DEFINE_FUNCTION

/// Functions of 1 argument do not require macros
template <typename ScalarType>
AutoDf<ScalarType> abs(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::abs(other);
}

template <typename ScalarType>
AutoDf<ScalarType> sin(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::sin(other);
}

template <typename ScalarType>
AutoDf<ScalarType> cos(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::cos(other);
}

#define AUTODF_INSTANTIATE_TYPE(typename)                                                      \
    template <>                                                                                \
    std::atomic<bool> AutoDf<typename>::create_variables_{true};                               \
    template <>                                                                                \
    std::atomic<AutoDf<typename>::IdType> AutoDf<typename>::id_increment_{0U};                 \
    template <>                                                                                \
    const std::shared_ptr<AutoDf<typename>::CallGraphNode> AutoDf<typename>::zero_ =           \
        std::make_shared<AutoDf<typename>::CallGraphNode>(AutoType::kConstType, typename(0.)); \
    AutoDf<typename> ____instantiate_AutoDf_##typename;

#ifndef AUTODF_DONT_INSTANTIATE_FLOAT
AUTODF_INSTANTIATE_TYPE(float);
#endif

#ifndef AUTODF_DONT_INSTANTIATE_DOUBLE
AUTODF_INSTANTIATE_TYPE(double);
#endif

/// By-default AutoDf is only instantiated for `float` and `double` types
/// In order to instantiate AutoDf for some other type, like std::complex you need to write in your code
/// AUTODF_INSTANTIATE_TYPE(std::complex);

template <typename ScalarType>
struct TerminationCriteria
{
    ScalarType expression_less_than = NAN;
    ScalarType step_less_than = 1e-6;
    ScalarType diff_less_than = 1e-8;
};

/// Gradient Descent minimization algorithm
template <typename ScalarType>
typename AutoDf<ScalarType>::Evaluation GradientDescent(
    const AutoDf<ScalarType>& minimize_expression,
    const TerminationCriteria<ScalarType> termination_criteria = {NAN, 1e-6, 1e-8},
    const ScalarType initial_step = ScalarType(0.01),
    const std::size_t max_iterations = 100U)
{
    auto prev_error = minimize_expression.eval();
    const bool termination_criteria_met = (!std::isnan(termination_criteria.expression_less_than)) &&
                                          (prev_error.value < termination_criteria.expression_less_than);

    if (std::isnan(prev_error.value) || termination_criteria_met)
    {
        return prev_error;
    }

    std::cout << "0: (initial) F[.]=" << prev_error.value << std::endl;

    // store previous variable values
    std::unordered_map<size_t, float> prev_values{};
    for (auto value_pair : minimize_expression.variables())
    {
        prev_values[value_pair.first] = *value_pair.second;
    }

    ScalarType current_step = initial_step;

    for (size_t iter = 1U; iter < max_iterations; iter++)
    {
        std::cout << iter << ": (rate=" << current_step << ") F[";

        for (auto value_pair : minimize_expression.variables())
        {
            const ScalarType derivative = prev_error.derivatives[value_pair.first];
            *value_pair.second = prev_values[value_pair.first] - derivative * current_step;
            std::cout << *value_pair.second << ",";
        }

        auto current_error = minimize_expression.eval();
        std::cout << "] = " << current_error.value << std::endl;

        const bool expression_criteria_met = (!std::isnan(termination_criteria.expression_less_than)) &&
                                             (current_error.value < termination_criteria.expression_less_than);

        ScalarType error_diff = prev_error.value - current_error.value;
        const bool diff_criteria_met =
            (!std::isnan(termination_criteria.diff_less_than)) && (error_diff < termination_criteria.diff_less_than);

        if (expression_criteria_met || diff_criteria_met)
        {
            return current_error;
        }

        // if error get worse -> "re-play" previous iteration with smaller step
        if (current_error.value > prev_error.value)
        {
            // reduce step
            current_step /= 2.;
        }
        else
        {

            float dot_value = 0.F, norm_value = 0.F;

            for (auto value_pair : minimize_expression.variables())
            {
                const ScalarType x_prev = prev_values[value_pair.first];
                const ScalarType dx_prev = prev_error.derivatives[value_pair.first];

                ScalarType x_curr = *value_pair.second;
                const ScalarType dx_curr = current_error.derivatives[value_pair.first];

                dot_value += (x_curr - x_prev) * (dx_curr - dx_prev);
                norm_value += (dx_curr - dx_prev) * (dx_curr - dx_prev);

                // update previous values
                prev_values[value_pair.first] = x_curr;
            }

            current_step = std::abs(dot_value) / norm_value;
            prev_error = current_error;
        }

        const bool step_criteria_met =
            (!std::isnan(termination_criteria.step_less_than)) && (current_step < termination_criteria.step_less_than);

        if (step_criteria_met)
        {
            // if current_step is getting too small -> restore previous values and return previuous error
            for (auto value_pair : minimize_expression.variables())
            {
                *value_pair.second = prev_values[value_pair.first];
            }
            return prev_error;
        }
    }

    // return best known error
    for (auto value_pair : minimize_expression.variables())
    {
        *value_pair.second = prev_values[value_pair.first];
    }
    return prev_error;
}
//
// template <typename ScalarType, std::size_t Size>
// typename AutoDf<ScalarType>::Evaluation LevenbergMarquardt(const std::array<AutoDf<ScalarType>, Size>& A_list,
//                                                          const std::array<ScalarType, Size>& b = {})
//{
//    // list all available variables
//    std::map<size_t, std::shared_ptr<ScalarType>> variables {};
//    std::for_each(A_list.begin(), A_list.end(), [&](auto& a){variables.insert(a.variables().begin());});
//
//}

}  // namespace tiny_autodf

#endif  // TINY_AUTODF_H_
