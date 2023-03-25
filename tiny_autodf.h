/*
 * This file is part of the Tiny-AutoDf distribution (https://github.com/sergehog/tiny_autodf)
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

template <typename ScalarType>
class AutoDf;

template <typename ScalarType>
std::ostream& operator<<(std::ostream& os, const AutoDf<ScalarType>& value);

template <typename ScalarType>
class AutoDf
{
  public:
    /// Each AutoDf despite its Type has ID of this type
    using IdType = size_t;

    /// Map of variables, allows their values to be modified
    using Variables = std::unordered_map<IdType, std::shared_ptr<ScalarType>>;

    /// Map of derivatives
    using Derivatives = std::unordered_map<IdType, ScalarType>;

    /// Type for storing result of AutoDf formula evaluation
    struct Evaluation
    {
        operator ScalarType() { return value; }
        ScalarType value;
        Derivatives derivatives;
    };

  private:
    /// List of all implemented AutoDf variants and operations
    enum class AutoDfType
    {
        kConstType,     /// just a const, cannot be changed
        kVariableType,  /// variable, may be assigned/changed
        kSumType,       /// result of sum of two input AutoDfs
        kSubtractType,  /// result of subtract of two input AutoDfs
        kMultType,      /// result of multiplication of two input AutoDfs
        kDivType,       /// result of division of two input AutoDfs
        kAbsType,       /// absolute value of one input AutoDf
        kMaxType,       /// maximum value among two input AutoDfs
        kMinType,       /// minimum value among two input AutoDfs
        kAtan2Type,     /// atan2(y,x)
        kSinType,       /// sin(x) value of one input AutoDf
        kCosType,       /// cos(x) value of one input AutoDf
        kSqrtType,      /// sqrt(x)
        kLogType,       /// log(x) natural logarithm
        kLog10Type,     /// log10(x) logarithm
        kExpType,       /// exp(x) -> e^x
        kPow2Type,      /// (x^2) same as (x*x) - square of x
        kReLUType,      /// ReLU(x) Rectified Linear Unit
        kLReLUType,     /// LReLU(x) Leaky Rectified Linear Unit
        kELUType,       /// ELU(x) Exponential Linear Unit
        // kPowType     /// pow(x, n) -> x^n
    };

    /// If bulk creation of Variables is enabled, it keeps corresponding value
    static std::atomic<bool> create_variables_;

    ///  Latest assigned ID value
    static std::atomic<IdType> id_increment_;

  protected:
    /// Structure for building and storing main Call Graph
    /// Each AutoDf has one corresponding shared_ptr CallGraphNode, but it's not necessary that each CallGraphNode
    /// instance has corresponding AutoDf
    struct CallGraphNode
    {
        IdType ID{};
        AutoDfType type{};
        size_t count{};
        std::string name;
        std::shared_ptr<CallGraphNode> left{};
        std::shared_ptr<CallGraphNode> right{};

        /// Current value of the Node (and corresponding AutoDf, if any)
        /// For constants and variables it's always their latest,
        /// For other types it's their last known value (updated after eval())
        std::shared_ptr<ScalarType> value{};

        /// List of values of AutoDf Variables, which current node depends on
        using InternalVariables = std::unordered_map<IdType, std::shared_ptr<ScalarType>>;
        InternalVariables variables{};

        CallGraphNode(const AutoDfType Type, const ScalarType& value_ = static_cast<ScalarType>(0.))
            : ID(id_increment_++), type(Type)
        {
            value = std::make_shared<ScalarType>(value_);
            count = 1;
        }

        /// Evaluation of call-graph
        /// @returns evaluated value & values of derivatives
        Evaluation eval()
        {
            switch (type)
            {
                case AutoDfType::kVariableType:
                    return evalVariable();
                case AutoDfType::kSumType:
                    return evalSum();
                case AutoDfType::kSubtractType:
                    return evalSubtract();
                case AutoDfType::kMultType:
                    return evalMult();
                case AutoDfType::kDivType:
                    return evalDiv();
                case AutoDfType::kAbsType:
                    return evalAbs();
                case AutoDfType::kMaxType:
                    return evalMax();
                case AutoDfType::kMinType:
                    return evalMin();
                case AutoDfType::kSinType:
                    return evalSin();
                case AutoDfType::kCosType:
                    return evalCos();
                case AutoDfType::kAtan2Type:
                    return evalAtan2();
                case AutoDfType::kSqrtType:
                    return evalSqrt();
                case AutoDfType::kLogType:
                    return evalLog();
                case AutoDfType::kLog10Type:
                    return evalLog10();
                case AutoDfType::kExpType:
                    return evalExp();
                case AutoDfType::kPow2Type:
                    return evalPow2();
                case AutoDfType::kReLUType:
                    return evalReLU();
                case AutoDfType::kLReLUType:
                    return evalLReLU();
                case AutoDfType::kELUType:
                    return evalELU();
                    // case AutoDfType::kPowType:
                    // return evalPow();
                case AutoDfType::kConstType:
                default:
                    return evalConst();
            }
        }

        Evaluation evalConst()
        {
            Evaluation result{*value};
            return result;
        }

        Evaluation evalVariable()
        {
            Evaluation result{*value};
            result.derivatives[ID] = ScalarType(1.);
            return result;
        }

        Evaluation evalAbs()
        {
            const auto left_eval = left->eval();
            Evaluation result{std::abs(left_eval.value)};
            *value = result.value;  // update the latest known value of this "dynamic" node
            const bool sign_changed = left_eval.value < ScalarType(0.);
            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = sign_changed ? -dx : dx;
            }
            return result;
        }

        Evaluation evalSin()
        {
            const auto left_eval = left->eval();
            Evaluation result{std::sin(left_eval.value)};
            *value = result.value;
            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = std::cos(left_eval.value) * dx;
            }
            return result;
        }

        Evaluation evalCos()
        {
            const auto left_eval = left->eval();
            Evaluation result{std::cos(left_eval.value)};
            *value = result.value;

            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = -std::sin(left_eval.value) * dx;
            }
            return result;
        }

        Evaluation evalSqrt()
        {
            const auto left_eval = left->eval();
            Evaluation result{std::sqrt(left_eval.value)};
            *value = result.value;

            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = ScalarType(0.5) * dx / result.value;
            }
            return result;
        }

        Evaluation evalLog()
        {
            const auto left_eval = left->eval();
            Evaluation result{std::log(left_eval.value)};
            *value = result.value;

            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = dx / left_eval.value;
            }
            return result;
        }

        Evaluation evalLog10()
        {
            const auto left_eval = left->eval();
            Evaluation result{std::log10(left_eval.value)};
            *value = result.value;

            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = dx / (left_eval.value * std::log(ScalarType(10.)));
            }
            return result;
        }

        Evaluation evalExp()
        {
            const auto left_eval = left->eval();
            Evaluation result{std::exp(left_eval.value)};
            *value = result.value;

            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = result.value * dx;
            }
            return result;
        }

        Evaluation evalPow2()
        {
            const auto left_eval = left->eval();
            Evaluation result{left_eval.value * left_eval.value};
            *value = result.value;

            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = ScalarType(2.) * left_eval.value * dx;
            }
            return result;
        }

        Evaluation evalReLU()
        {
            const auto left_eval = left->eval();
            Evaluation result{left_eval.value > ScalarType(0.) ? left_eval.value : ScalarType(0.)};
            *value = result.value;

            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = left_eval.value > ScalarType(0.) ? dx : ScalarType(0.F);
            }
            return result;
        }

        Evaluation evalLReLU()
        {
            const auto left_eval = left->eval();
            Evaluation result{left_eval.value > ScalarType(0.) ? left_eval.value : ScalarType(0.01) * left_eval.value};
            *value = result.value;

            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] = left_eval.value > ScalarType(0.) ? dx : ScalarType(0.01) * dx;
            }
            return result;
        }

        Evaluation evalELU()
        {
            const auto left_eval = left->eval();
            Evaluation result{left_eval.value > ScalarType(0.)
                                  ? left_eval.value
                                  : ScalarType(0.01) * (std::exp(left_eval.value) - ScalarType(1.))};
            *value = result.value;

            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] =
                    left_eval.value > ScalarType(0.) ? dx : ScalarType(0.01) * std::exp(left_eval.value) * dx;
            }
            return result;
        }

        //        Evaluation evalPow()
        //        {
        //            const auto left_eval = left->eval();
        //            Evaluation result{std::exp(left_eval.value)};
        //            *value = result.value;
        //
        //            for (auto dx_iter : left_eval.derivatives)
        //            {
        //                const IdType id = dx_iter.first;
        //                const ScalarType dx = dx_iter.second;
        //                result.derivatives[id] = result.value * dx;
        //            }
        //            return result;
        //        }

        Evaluation evalSum()
        {
            const auto left_eval = left->eval();
            const auto right_eval = right->eval();
            Evaluation result{left_eval.value + right_eval.value};
            *value = result.value;
            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] += dx;
            }
            for (auto dx_iter : right_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] += dx;
            }

            return result;
        }

        Evaluation evalSubtract()
        {
            const auto left_eval = left->eval();
            const auto right_eval = right->eval();
            Evaluation result{left_eval.value - right_eval.value};
            *value = result.value;
            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] += dx;
            }
            for (auto dx_iter : right_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] -= dx;
            }

            return result;
        }

        Evaluation evalMult()
        {
            const auto left_eval = left->eval();
            const auto right_eval = right->eval();
            Evaluation result{left_eval.value * right_eval.value, {}};
            *value = result.value;
            for (auto dx_iter : left_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] += dx * right_eval.value;
            }
            for (auto dx_iter : right_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] += dx * left_eval.value;
            }

            return result;
        }

        Evaluation evalAtan2()
        {
            const auto y_eval = left->eval();
            const auto x_eval = right->eval();
            const ScalarType y = y_eval.value;
            const ScalarType x = x_eval.value;
            Evaluation result{std::atan2(y, x), {}};
            *value = result.value;

            // df(..)/dy(t) = x(k) / (x^2(k) + y^2(t))
            for (auto dy_iter : y_eval.derivatives)  // dy
            {
                const IdType id = dy_iter.first;
                const ScalarType dy = dy_iter.second;
                result.derivatives[id] += dy * x / (y * y + x * x);
            }

            // df(..)/dx(k) = - y(t) / (x^2(k) - y^2(t))
            for (auto dx_iter : x_eval.derivatives)
            {
                const IdType id = dx_iter.first;
                const ScalarType dx = dx_iter.second;
                result.derivatives[id] -= dx * y / (y * y + x * x);
            }
            return result;
        }

        /// For explanation please refer to
        /// https://www.mathsisfun.com/calculus/derivatives-rules.html
        Evaluation evalDiv()
        {
            const auto left_eval = left->eval();
            const auto right_eval = right->eval();
            Evaluation result{left_eval.value / right_eval.value};
            *value = result.value;
            const auto f = left_eval.value;
            const auto g = right_eval.value;
            for (auto df_iter : left_eval.derivatives)
            {
                const IdType id = df_iter.first;
                const ScalarType df = df_iter.second;
                result.derivatives[id] += df * g / (g * g);
            }
            for (auto dg_iter : right_eval.derivatives)
            {
                const IdType id = dg_iter.first;
                const ScalarType dg = dg_iter.second;
                result.derivatives[id] -= dg * f / (g * g);
            }

            return result;
        }

        Evaluation evalMax()
        {
            const auto left_eval = left->eval();
            const auto right_eval = right->eval();
            Evaluation result{std::max(left_eval.value, right_eval.value)};
            *value = result.value;
            if (left_eval.value >= right_eval.value)
            {
                for (auto dx_iter : left_eval.derivatives)
                {
                    const IdType id = dx_iter.first;
                    const ScalarType dx = dx_iter.second;
                    result.derivatives[id] = dx;
                }
            }
            else
            {
                for (auto dx_iter : right_eval.derivatives)
                {
                    const IdType id = dx_iter.first;
                    const ScalarType dx = dx_iter.second;
                    result.derivatives[id] += dx;
                }
            }

            return result;
        }

        Evaluation evalMin()
        {
            const auto left_eval = left->eval();
            const auto right_eval = right->eval();
            Evaluation result{std::min(left_eval.value, right_eval.value)};
            *value = result.value;
            if (left_eval.value <= right_eval.value)
            {
                for (auto dx_iter : left_eval.derivatives)
                {
                    const IdType id = dx_iter.first;
                    const ScalarType dx = dx_iter.second;
                    result.derivatives[id] = dx;
                }
            }
            else
            {
                for (auto dx_iter : right_eval.derivatives)
                {
                    const IdType id = dx_iter.first;
                    const ScalarType dx = dx_iter.second;
                    result.derivatives[id] += dx;
                }
            }

            return result;
        }

        std::ostream& print(std::ostream& stream)
        {
            switch (type)
            {
                case AutoDfType::kVariableType:
                    stream << name;
                    break;
                case AutoDfType::kSumType:
                    stream << "(";
                    left->print(stream) << " + ";
                    right->print(stream) << ")";
                    break;
                case AutoDfType::kSubtractType:
                    stream << "(";
                    left->print(stream) << " - ";
                    right->print(stream) << ")";
                    break;
                case AutoDfType::kMultType:
                    left->print(stream) << "*";
                    right->print(stream);
                    break;
                case AutoDfType::kDivType:
                    stream << "(";
                    left->print(stream) << ") / (";
                    right->print(stream) << ")";
                    break;
                case AutoDfType::kAbsType:
                    stream << "abs(";
                    left->print(stream) << ")";
                    break;
                case AutoDfType::kMaxType:
                    stream << "max(";
                    left->print(stream) << ",";
                    right->print(stream) << ")";
                    break;
                case AutoDfType::kMinType:
                    stream << "min(";
                    left->print(stream) << ",";
                    right->print(stream) << ")";
                    break;
                case AutoDfType::kSinType:
                    stream << "sin(";
                    left->print(stream) << ")";
                    break;
                case AutoDfType::kCosType:
                    stream << "cos(";
                    left->print(stream) << ")";
                    break;
                case AutoDfType::kAtan2Type:
                    stream << "atan2(";
                    left->print(stream) << ",";
                    right->print(stream) << ")";
                    break;
                case AutoDfType::kSqrtType:
                    stream << "sqrt(";
                    left->print(stream) << ")";
                    break;
                case AutoDfType::kLogType:
                    stream << "log(";
                    left->print(stream) << ")";
                    break;
                case AutoDfType::kLog10Type:
                    stream << "log10(";
                    left->print(stream) << ")";
                    break;
                case AutoDfType::kExpType:
                    stream << "exp(";
                    left->print(stream) << ")";
                    break;
                case AutoDfType::kPow2Type:
                    stream << "(";
                    left->print(stream) << ")^2";
                    break;
                case AutoDfType::kReLUType:
                    stream << "relu(";
                    left->print(stream) << ")";
                    break;
                case AutoDfType::kLReLUType:
                    stream << "lrelu(";
                    left->print(stream) << ")";
                    break;
                case AutoDfType::kELUType:
                    stream << "elu(";
                    left->print(stream) << ")";
                    break;

                case AutoDfType::kConstType:
                default:
                    stream << *value;
            }
            return stream;
        }
    };

    /// Graph Node belonging to current variable / AutoDf instance
    std::shared_ptr<CallGraphNode> node_{};

  private:
    /// In order not to multiply zero-valued ConstType nodes, we maintain single such node and re-use
    /// Reduces memory-consumption as well unpredictable id_increment grow
    static const std::shared_ptr<CallGraphNode> zero_;

    /// Creates AutoDf of specified type, and takes their ownership if needed
    AutoDf(const AutoDfType type,
           const std::shared_ptr<CallGraphNode>& left,
           const std::shared_ptr<CallGraphNode>& right,
           const ScalarType& value = static_cast<ScalarType>(0.))
    {
        node_ = std::make_shared<CallGraphNode>(type, value);
        node_->left = left;
        node_->right = right;
        node_->count = 0U;  // ToDo: shall it be 1 here?
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

    /// private Ctor for manual node setup
    explicit AutoDf(const std::shared_ptr<CallGraphNode>& node_in) : node_(node_in) {}

  public:
    /// Read-only ID value, could be used to distinguish partial derivative of this variable
    IdType ID() const { return node_->ID; }

    /// @returns number of nodes in the call-graph of this AutoDf
    size_t count() const { return node_->count; }

    std::string& name() { return node_->name; }

    std::string name() const { return node_->name; }

    /// All AutoDfs created with default constructor will be Consts after calling this function
    static void ConstantsByDefault() { create_variables_ = false; }

    /// All AutoDfs, created with default constructor will be Variables after calling this function
    /// This behaviour is enabled by-default
    static void VariablesByDefault() { create_variables_ = true; }

    /// Creates kVariableType or kConstType (depending on default_type_) with zero value
    AutoDf()
    {
        if (create_variables_)
        {
            node_ = std::make_shared<CallGraphNode>(AutoDfType::kVariableType);
            node_->variables[node_->ID] = node_->value;
            node_->name = "var" + std::to_string(node_->ID);
        }
        else
        {
            node_ = zero_;
        }
    }

    /// Creates named Variable
    explicit AutoDf(const std::string name, ScalarType value = ScalarType(0.0))
    {
        node_ = std::make_shared<CallGraphNode>(AutoDfType::kVariableType);
        *(node_->value) = value;
        node_->variables[node_->ID] = node_->value;
        node_->name = name;
    }

    /// Constructor with initializer_list always creates Variable
    AutoDf(std::initializer_list<ScalarType> list)
    {
        node_ = std::make_shared<CallGraphNode>(AutoDfType::kVariableType);
        node_->variables[node_->ID] = node_->value;
        if (list.size() > 0)
        {
            auto it = list.begin();
            *node_->value = ScalarType(*it);
        }

        node_->name = "var" + std::to_string(node_->ID);
    }

    /// Ctor with underlying type valued argument
    /// Creates kVariableType or kConstType (depending on create_variables_) with given scalar value
    AutoDf(const ScalarType& scalar)
    {
        if (create_variables_)
        {
            node_ = std::make_shared<CallGraphNode>(AutoDfType::kVariableType, scalar);
            node_->variables[node_->ID] = node_->value;
        }
        else
        {
            if (scalar == static_cast<ScalarType>(0.0))
            {
                // avoid creation trivial const type node, by re-using zero-valued one
                node_ = zero_;
            }
            else
            {
                node_ = std::make_shared<CallGraphNode>(AutoDfType::kConstType, scalar);
            }
        }
    }

    /// Copy-Constructor. Re-uses call-graph node
    AutoDf(const AutoDf<ScalarType>& other) { node_ = other.node_; }

    /// Additional way to create AutoDf of required type
    AutoDf(const ScalarType& value, const bool non_mutable)
    {
        const auto type = (non_mutable ? AutoDfType::kConstType : AutoDfType::kVariableType);
        node_ = std::make_shared<CallGraphNode>(type, value);
        if (node_->type == AutoDfType::kVariableType)
        {
            node_->variables[node_->ID] = node_->value;
        }
    }

    /// Assignment operator for underlying type value
    AutoDf& operator=(const ScalarType& scalar)
    {
        if (node_ == zero_ && scalar != ScalarType(0))
        {
            // no longer may use zero_, so create new Const node
            node_ = std::make_shared<CallGraphNode>(AutoDfType::kConstType, scalar);
        }
        else if (node_->type == AutoDfType::kConstType || node_->type == AutoDfType::kVariableType)
        {
            *node_->value = scalar;
        }
        return *this;
    }

    /// Copy-Assignment. Re-uses call-graph node
    AutoDf<ScalarType>& operator=(const AutoDf<ScalarType>& other)
    {
        node_ = other.node_;
        return *this;
    }

    /// @returns List of values of AutoDf Variables contributing to this equation (AutoDf value), if any
    Variables variables() const { return node_->variables; }

    /// Evaluates current value and gradients of the AutoDf for current state of call-graph and Variable values
    Evaluation eval() const { return node_->eval(); }

    /// Returns current value
    ScalarType operator()() const { return *node_->value; }

    /// @returns latest calculated value or sets new one (but only for variables)
    /// One may change underlying value directly
    ScalarType& value() const
    {
        if (node_->type == AutoDfType::kVariableType)
        {
            return *node_->value;
        }
        else
        {
            // prohibit exposing real value if this node is not of the variable type
            // however we may not expose local variable either, so we have some static fake thingy
            static ScalarType fake_value = ScalarType(0.0);
            fake_value = *node_->value;
            return fake_value;
        }
    }

    /// Another way to obtain current node value
    explicit operator ScalarType() const { return *node_->value; }

    /// In-place addition with underlying type argument
    AutoDf<ScalarType>& operator+=(const ScalarType value)
    {
        if (value == static_cast<ScalarType>(0.0))
        {
            // avoid creation graph nodes, when not really needed
            // (adding with const zero wont change anything)
            return *this;
        }
        else if (node_ == zero_)
        {
            // no longer may re-use zero_ node, so have to create specific const node
            node_ = std::make_shared<CallGraphNode>(AutoDfType::kConstType, value);
            return *this;
        }
        else if (node_->type == AutoDfType::kConstType)
        {
            const ScalarType old_node_value = *node_->value;
            node_ = std::make_shared<CallGraphNode>(AutoDfType::kConstType, old_node_value + value);
            return *this;
        }

        AutoDf<ScalarType> other(value, true);
        return InPlaceOperator(AutoDfType::kSumType, other.node_, *node_->value + *other.node_->value);
    }

    AutoDf<ScalarType>& operator+=(const AutoDf<ScalarType>& other)
    {
        if (other.node_->type == AutoDfType::kConstType && *other.node_->value == static_cast<ScalarType>(0))
        {
            // avoid creation graph nodes, when not really needed
            return *this;
        }
        else if (node_->type == AutoDfType::kConstType && other.node_->type == AutoDfType::kConstType)
        {
            const ScalarType node_value = *node_->value;
            node_ = std::make_shared<CallGraphNode>(AutoDfType::kConstType, node_value + *other.node_->value);
            return *this;
        }

        return InPlaceOperator(AutoDfType::kSumType, other.node_, *node_->value + *other.node_->value);
    }

    AutoDf<ScalarType>& operator-=(const ScalarType value)
    {
        if (value == static_cast<ScalarType>(0))
        {
            // avoid creating graph nodes, when not really needed
            return *this;
        }
        else if (node_->type == AutoDfType::kConstType)
        {
            const ScalarType node_value = *node_->value;
            node_ = std::make_shared<CallGraphNode>(AutoDfType::kConstType, node_value - value);
            return *this;
        }

        AutoDf<ScalarType> other(value, true);
        return InPlaceOperator(AutoDfType::kSubtractType, other.node_, *node_->value - *other.node_->value);
    }

    AutoDf<ScalarType>& operator-=(const AutoDf<ScalarType>& other)
    {
        if (other.node_->type == AutoDfType::kConstType && *other.node_->value == static_cast<ScalarType>(0))
        {
            return *this;
        }
        else if (node_->type == AutoDfType::kConstType && other.node_->type == AutoDfType::kConstType)
        {
            const ScalarType node_value = *node_->value;
            node_ = std::make_shared<CallGraphNode>(AutoDfType::kConstType, node_value - *other.node_->value);
            return *this;
        }

        return InPlaceOperator(AutoDfType::kSubtractType, other.node_, *node_->value - *other.node_->value);
    }

    static AutoDf<ScalarType> abs(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoDfType::kAbsType, other.node_, nullptr, std::abs(*other.node_->value));
    }

    static AutoDf<ScalarType> min(const AutoDf<ScalarType>& left, const AutoDf<ScalarType>& right)
    {
        return AutoDf<ScalarType>(
            AutoDfType::kMinType, left.node_, right.node_, std::min(*left.node_->value, *right.node_->value));
    }

    static AutoDf<ScalarType> max(const AutoDf<ScalarType>& left, const AutoDf<ScalarType>& right)
    {
        return AutoDf<ScalarType>(
            AutoDfType::kMaxType, left.node_, right.node_, std::max(*left.node_->value, *right.node_->value));
    }

    static AutoDf<ScalarType> sin(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoDfType::kSinType, other.node_, nullptr, std::sin(*other.node_->value));
    }

    static AutoDf<ScalarType> cos(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoDfType::kCosType, other.node_, nullptr, std::cos(*other.node_->value));
    }

    static AutoDf<ScalarType> sqrt(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoDfType::kSqrtType, other.node_, nullptr, std::sqrt(*other.node_->value));
    }

    static AutoDf<ScalarType> log(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoDfType::kLogType, other.node_, nullptr, std::log(*other.node_->value));
    }

    static AutoDf<ScalarType> log10(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoDfType::kLog10Type, other.node_, nullptr, std::log10(*other.node_->value));
    }

    static AutoDf<ScalarType> exp(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(AutoDfType::kExpType, other.node_, nullptr, std::exp(*other.node_->value));
    }

    static AutoDf<ScalarType> pow2(const AutoDf<ScalarType>& other)
    {
        return AutoDf<ScalarType>(
            AutoDfType::kPow2Type, other.node_, nullptr, (*other.node_->value) * (*other.node_->value));
    }

    static AutoDf<ScalarType> relu(const AutoDf<ScalarType>& other)
    {
        const ScalarType x = (*other.node_->value);
        const ScalarType value = (std::max(x, ScalarType(0.)));
        return AutoDf<ScalarType>(AutoDfType::kReLUType, other.node_, nullptr, value);
    }

    static AutoDf<ScalarType> lrelu(const AutoDf<ScalarType>& other)
    {
        const ScalarType x = (*other.node_->value);
        const ScalarType value = (x > ScalarType(0.) ? x : ScalarType(0.01) * x);
        return AutoDf<ScalarType>(AutoDfType::kReLUType, other.node_, nullptr, value);
    }

    static AutoDf<ScalarType> elu(const AutoDf<ScalarType>& other)
    {
        const ScalarType x = (*other.node_->value);
        const ScalarType value = (x > ScalarType(0.) ? x : ScalarType(0.01) * (std::exp(x) - ScalarType(1.)));
        return AutoDf<ScalarType>(AutoDfType::kReLUType, other.node_, nullptr, value);
    }

    static AutoDf<ScalarType> atan2(const AutoDf<ScalarType>& y, const AutoDf<ScalarType>& x)
    {
        const ScalarType value = std::atan2(*y.node_->value, *x.node_->value);
        return AutoDf<ScalarType>(AutoDfType::kAtan2Type, y.node_, x.node_, value);
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
    AutoDf<ScalarType>& InPlaceOperator(const AutoDfType& type,
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
        if (left->type == AutoDfType::kConstType && *left->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(right);
        }
        if (right->type == AutoDfType::kConstType && *right->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(left);
        }

        const ScalarType value = *left->value + *right->value;
        return AutoDf<ScalarType>(AutoDfType::kSumType, left, right, value);
    }

    static AutoDf<ScalarType> make_sub(const std::shared_ptr<CallGraphNode>& left,
                                       const std::shared_ptr<CallGraphNode>& right)
    {
        if (right->type == AutoDfType::kConstType && *right->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(left);
        }

        const ScalarType value = *left->value - *right->value;
        return AutoDf<ScalarType>(AutoDfType::kSubtractType, left, right, value);
    }

    static AutoDf<ScalarType> make_mult(const std::shared_ptr<CallGraphNode>& left,
                                        const std::shared_ptr<CallGraphNode>& right)
    {
        if (left->type == AutoDfType::kConstType && *left->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(zero_);
        }
        if (right->type == AutoDfType::kConstType && *right->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(zero_);
        }
        if (left->type == AutoDfType::kConstType && *left->value == static_cast<ScalarType>(1))
        {
            return AutoDf<ScalarType>(right);
        }
        if (right->type == AutoDfType::kConstType && *right->value == static_cast<ScalarType>(1))
        {
            return AutoDf<ScalarType>(left);
        }

        const ScalarType value = *left->value * *right->value;
        return AutoDf<ScalarType>(AutoDfType::kMultType, left, right, value);
    }

    static AutoDf<ScalarType> make_div(const std::shared_ptr<CallGraphNode>& left,
                                       const std::shared_ptr<CallGraphNode>& right)
    {
        if (left->type == AutoDfType::kConstType && *left->value == static_cast<ScalarType>(0))
        {
            return AutoDf<ScalarType>(zero_);
        }
        if (right->type == AutoDfType::kConstType && *right->value == static_cast<ScalarType>(1))
        {
            return AutoDf<ScalarType>(left);
        }

        const ScalarType value = *left->value / *right->value;
        return AutoDf<ScalarType>(AutoDfType::kDivType, left, right, value);
    }

  public:
    friend std::ostream& operator<<<>(std::ostream& os, const AutoDf<ScalarType>& value);
};

template <>
std::ostream& operator<<<>(std::ostream& os, const AutoDf<float>& value)
{
    return value.node_->print(os);
}

template <typename ScalarType>
AutoDf<ScalarType> operator-(AutoDf<ScalarType> const& other)
{
    if (other.node_->type == AutoDf<ScalarType>::AutoDfType::kConstType)
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
AUTODF_DEFINE_FUNCTION(atan2);
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

template <typename ScalarType>
AutoDf<ScalarType> sqrt(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::sqrt(other);
}

template <typename ScalarType>
AutoDf<ScalarType> log(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::log(other);
}

template <typename ScalarType>
AutoDf<ScalarType> log10(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::log10(other);
}

template <typename ScalarType>
AutoDf<ScalarType> exp(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::exp(other);
}

template <typename ScalarType>
AutoDf<ScalarType> pow2(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::pow2(other);
}

template <typename ScalarType>
AutoDf<ScalarType> relu(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::relu(other);
}

template <typename ScalarType>
AutoDf<ScalarType> lrelu(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::lrelu(other);
}

template <typename ScalarType>
AutoDf<ScalarType> elu(const AutoDf<ScalarType>& other)
{
    return AutoDf<ScalarType>::elu(other);
}

#define AUTODF_INSTANTIATE_TYPE(typename)                                                        \
    template <>                                                                                  \
    std::atomic<bool> AutoDf<typename>::create_variables_{true};                                 \
    template <>                                                                                  \
    std::atomic<AutoDf<typename>::IdType> AutoDf<typename>::id_increment_{0U};                   \
    template <>                                                                                  \
    const std::shared_ptr<AutoDf<typename>::CallGraphNode> AutoDf<typename>::zero_ =             \
        std::make_shared<AutoDf<typename>::CallGraphNode>(AutoDfType::kConstType, typename(0.)); \
    AutoDf<typename> ____instantiate_AutoDf_##typename;
// friend std::ostream& operator<< <>(std::ostream& os, const AutoDf<typename> & value)

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
/// @param minimize_expression AutoDf<> with the equation which need to be minimized
/// @param termination_criteria
/// @param initial_step aka learning rate, automatically adjusted
/// @param max_iterations
template <typename ScalarType>
typename AutoDf<ScalarType>::Evaluation GradientDescent(
    const AutoDf<ScalarType>& minimize_expression,
    const TerminationCriteria<ScalarType> termination_criteria = {NAN, 1e-6, 1e-8},
    const ScalarType initial_step = ScalarType(0.01),
    const std::size_t max_iterations = 100U)
{
    using IdType = typename AutoDf<ScalarType>::IdType;
    auto previous_error = minimize_expression.eval();
    const bool termination_criteria_met = (!std::isnan(termination_criteria.expression_less_than)) &&
                                          (previous_error.value < termination_criteria.expression_less_than);

    if (std::isnan(previous_error.value) || termination_criteria_met)
    {
        return previous_error;
    }
#ifdef AUTODF_DEBUG_OUTPUT
    std::cout << "0: (initial) F[.]=" << previous_error.value << std::endl;
#endif
    // store previous variable values
    std::unordered_map<IdType, float> previous_values{};
    for (auto value_pair : minimize_expression.variables())
    {
        previous_values[value_pair.first] = *value_pair.second;
    }

    // adjustable learning rate
    ScalarType current_step = initial_step;

    for (size_t iter = 1U; iter < max_iterations; iter++)
    {
#ifdef AUTODF_DEBUG_OUTPUT
        std::cout << iter << ": (rate=" << current_step << ") F[";
#endif
        // update current values
        for (auto value_pair : minimize_expression.variables())
        {
            const ScalarType derivative = previous_error.derivatives[value_pair.first];
            *value_pair.second = previous_values[value_pair.first] - derivative * current_step;
#ifdef AUTODF_DEBUG_OUTPUT
            std::cout << *value_pair.second << ",";
#endif
        }

        // evaluate
        auto current_error = minimize_expression.eval();
#ifdef AUTODF_DEBUG_OUTPUT
        std::cout << "] = " << current_error.value << std::endl;
#endif
        const bool expression_criteria_met = (!std::isnan(termination_criteria.expression_less_than)) &&
                                             (current_error.value < termination_criteria.expression_less_than);

        ScalarType error_diff = previous_error.value - current_error.value;
        const bool diff_criteria_met = (!std::isnan(termination_criteria.diff_less_than)) && (error_diff > 0.) &&
                                       (error_diff < termination_criteria.diff_less_than);

        if (expression_criteria_met || diff_criteria_met)
        {
            return current_error;
        }

        // if error get worse -> restore previous values, and try again with smaller step
        if (std::isnan(current_error.value) || current_error.value > previous_error.value)
        {
            // restore values
            for (auto value_pair : minimize_expression.variables())
            {
                *value_pair.second = previous_values[value_pair.first];
            }

            // reduce step
            current_step /= 2.;

            const bool step_criteria_met =
                (std::isnan(current_step)) || ((!std::isnan(termination_criteria.step_less_than)) &&
                                               (current_step < termination_criteria.step_less_than));

            if (step_criteria_met)
            {
                return previous_error;
            }

            continue;
        }
        else
        {
            float dot_value = 0.F, norm_value = 0.F;

            for (auto value_pair : minimize_expression.variables())
            {
                const ScalarType x_prev = previous_values[value_pair.first];
                const ScalarType dx_prev = previous_error.derivatives[value_pair.first];

                ScalarType x_curr = *value_pair.second;
                const ScalarType dx_curr = current_error.derivatives[value_pair.first];

                dot_value += (x_curr - x_prev) * (dx_curr - dx_prev);
                norm_value += (dx_curr - dx_prev) * (dx_curr - dx_prev);

                // update previous values
                previous_values[value_pair.first] = x_curr;
            }

            current_step = std::abs(dot_value) / norm_value;
            previous_error = current_error;
        }
    }

    // return best known error
    for (auto value_pair : minimize_expression.variables())
    {
        *value_pair.second = previous_values[value_pair.first];
    }
    return previous_error;
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
