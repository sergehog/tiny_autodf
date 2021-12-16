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

#include "../tiny_autodf.h"
#include <gtest/gtest.h>

using Float = tiny_autodf::AutoDf<float>;

TEST(AutoDfTest, BasicTest)
{
    Float x("x");
    Float y = x * x + 2.F * x + 5.F;
    EXPECT_EQ(x.value(), 0.f);
    EXPECT_EQ(y.value(), 5.f);
    std::cout << y << "=" << y() << std::endl;
    x = 1;
    EXPECT_EQ(float(y.eval()), 8.f);
    auto z = 0.F - abs(max(y / x, min(sin(x), cos(x))));
    std::cout << "z=" << z << " = " << z() << std::endl;
}

TEST(AutoDfTest, OneDependentVariableTest)
{
    Float x = 15.F;
    Float y = x + 5.F;
    Float z = (2.f * x + 2.F) * (y - 3.F);
    Float w = z / (x + 1.F);

    EXPECT_EQ(x.value(), 15.f);
    EXPECT_EQ(y.value(), 20.f);
    EXPECT_EQ(z.value(), 544.f);
    EXPECT_EQ(w.value(), 544.f / 16.f);

    ASSERT_EQ(x.variables().size(), 1);
    ASSERT_EQ(y.variables().size(), 1);
    ASSERT_EQ(z.variables().size(), 1);
    ASSERT_EQ(w.variables().size(), 1);

    y += 5.f;
    EXPECT_EQ(y.value(), 25.f);
    auto xe = x.eval();
    auto ye = y.eval();
    auto ze = z.eval();
    auto we = w.eval();

    EXPECT_EQ(x.value(), xe.value);
    EXPECT_EQ(y.value(), ye.value);
    EXPECT_EQ(z.value(), ze.value);
    EXPECT_EQ(w.value(), we.value);

    ASSERT_EQ(xe.derivatives.size(), 1);
    ASSERT_EQ(ye.derivatives.size(), 1);
    ASSERT_EQ(ze.derivatives.size(), 1);
    ASSERT_EQ(we.derivatives.size(), 1);

    EXPECT_EQ(xe.derivatives[x.ID()], 1.F);
    EXPECT_EQ(ye.derivatives[x.ID()], 1.F);
    EXPECT_EQ(ze.derivatives[x.ID()], 66.F);
    EXPECT_EQ(we.derivatives[x.ID()], 2.F);

    // nothing changed, except y
    EXPECT_EQ(x.value(), 15.f);
    EXPECT_EQ(y.value(), 25.f);
    EXPECT_EQ(z.value(), 544.f);
    EXPECT_EQ(w.value(), 544.f / 16.f);

    x.value() += 1.F;
    // re-calculate only one formula, but other are updated too, since they are part of call-graph
    w.eval();
    EXPECT_EQ(x.value(), 16.f);
    EXPECT_EQ(y.value(), 25.f);
    EXPECT_EQ(z.value(), 612.f);
    EXPECT_EQ(w.value(), 36.f);
}

TEST(AutoDfTest, SumTest)
{
    Float x = 7.F;
    Float y = (x + 3.F) + 5.F;
    Float z = (5.F + y) + x;

    EXPECT_EQ(x.value(), 7.F);
    EXPECT_EQ(y.value(), 15.F);
    EXPECT_EQ(z.value(), 27.F);

    ASSERT_EQ(x.variables().size(), 1);
    ASSERT_EQ(y.variables().size(), 1);
    ASSERT_EQ(z.variables().size(), 1);

    x.value() -= 1.F;
    // x value changes, but not others
    EXPECT_EQ(x.value(), 6.F);
    EXPECT_EQ(y.value(), 15.F);
    EXPECT_EQ(z.value(), 27.F);

    // re-evaluate
    auto xe = x.eval();
    auto ye = y.eval();
    auto ze = z.eval();

    // now all values changed
    EXPECT_EQ(x.value(), 6.F);
    EXPECT_EQ(y.value(), 14.F);
    EXPECT_EQ(z.value(), 25.F);

    // evaluation results are the same
    EXPECT_EQ(x.value(), xe.value);
    EXPECT_EQ(y.value(), ye.value);
    EXPECT_EQ(z.value(), ze.value);

    // number of partial derivatives is 1
    ASSERT_EQ(xe.derivatives.size(), 1);
    ASSERT_EQ(ye.derivatives.size(), 1);
    ASSERT_EQ(ze.derivatives.size(), 1);

    // all partial derivatives are 1, since only + was used
    EXPECT_EQ(xe.derivatives[x.ID()], 1.F);
    EXPECT_EQ(ye.derivatives[x.ID()], 1.F);
    EXPECT_EQ(ze.derivatives[x.ID()], 2.F);
}

TEST(AutoDfTest, SubtractTest)
{
    Float x = 10.F;
    Float y = 20.F - x - 5.F;
    Float z = 7.F - (10.F - y);

    EXPECT_EQ(x.value(), 10.f);
    EXPECT_EQ(y.value(), 5.F);
    EXPECT_EQ(z.value(), 2.F);

    ASSERT_EQ(x.variables().size(), 1);
    ASSERT_EQ(y.variables().size(), 1);
    ASSERT_EQ(z.variables().size(), 1);

    x.value() -= 1.F;
    // x value changes, but not others
    EXPECT_EQ(x.value(), 9.f);
    EXPECT_EQ(y.value(), 5.F);
    EXPECT_EQ(z.value(), 2.F);

    // re-evaluate
    auto xe = x.eval();
    auto ye = y.eval();
    auto ze = z.eval();

    // x value changes, but not others
    EXPECT_EQ(x.value(), 9.f);
    EXPECT_EQ(y.value(), 6.F);
    EXPECT_EQ(z.value(), 3.F);

    // evaluation results are the same
    EXPECT_EQ(x.value(), xe.value);
    EXPECT_EQ(y.value(), ye.value);
    EXPECT_EQ(z.value(), ze.value);

    // number of partial derivatives is 1
    ASSERT_EQ(xe.derivatives.size(), 1);
    ASSERT_EQ(ye.derivatives.size(), 1);
    ASSERT_EQ(ze.derivatives.size(), 1);

    // all partial derivatives are 1, since only + was used
    EXPECT_EQ(xe.derivatives[x.ID()], 1.F);
    EXPECT_EQ(ye.derivatives[x.ID()], -1.F);
    EXPECT_EQ(ze.derivatives[x.ID()], -1.F);
}

TEST(AutoDfTest, MultiplicationTest)
{
    Float x = 7.F;
    Float y = (x - 1.f) * (x + 1.F) * 2.F;

    EXPECT_EQ(x.value(), 7.f);
    EXPECT_EQ(y.value(), 6.f * 8.F * 2.F);

    ASSERT_EQ(x.variables().size(), 1);
    ASSERT_EQ(y.variables().size(), 1);

    x.value() -= 1.F;
    // x value changes, but not others
    EXPECT_EQ(x.value(), 6.f);
    EXPECT_EQ(y.value(), 6.f * 8.F * 2.F);

    // re-evaluate
    auto xe = x.eval();
    auto ye = y.eval();

    // x value changes, but not others
    EXPECT_EQ(x.value(), 6.f);
    EXPECT_EQ(y.value(), 5.f * 7.F * 2.F);

    // evaluation results are the same
    EXPECT_EQ(x.value(), xe.value);
    EXPECT_EQ(y.value(), ye.value);

    // number of partial derivatives is 1
    ASSERT_EQ(xe.derivatives.size(), 1);
    ASSERT_EQ(ye.derivatives.size(), 1);

    // all partial derivatives are 1, since only + was used
    EXPECT_EQ(xe.derivatives[x.ID()], 1.F);
    EXPECT_EQ(ye.derivatives[x.ID()], 4 * x.value());
}

TEST(AutoDfTest, DivisionTest)
{
    Float x = 7.F;
    Float y = (x - 1.f) / (x + 1.F) / 2.F;

    EXPECT_EQ(x.value(), 7.f);
    EXPECT_EQ(y.value(), 6.f / 8.F / 2.F);

    ASSERT_EQ(x.variables().size(), 1);
    ASSERT_EQ(y.variables().size(), 1);

    x.value() -= 1.F;
    // x value changes, but not others
    EXPECT_EQ(x.value(), 6.f);
    EXPECT_EQ(y.value(), 6.f / 8.F / 2.F);

    // re-evaluate
    auto xe = x.eval();
    auto ye = y.eval();

    // x value changes, but not others
    EXPECT_EQ(x.value(), 6.f);
    EXPECT_EQ(y.value(), 5.f / 7.F / 2.F);

    // evaluation results are the same
    EXPECT_EQ(x.value(), xe.value);
    EXPECT_EQ(y.value(), ye.value);

    // number of partial derivatives is 1
    ASSERT_EQ(xe.derivatives.size(), 1);
    ASSERT_EQ(ye.derivatives.size(), 1);

    // all partial derivatives are 1, since only + was used
    EXPECT_EQ(xe.derivatives[x.ID()], 1.F);
    EXPECT_NEAR(ye.derivatives[x.ID()], 0.0204082F, 0.00001F);
}

TEST(AutoDfTest, AbsMinMaxTest)
{
    Float x = 7.F;
    Float y = -5.F;
    Float absx = abs(x);
    Float absy = abs(y);
    Float minxy = min(x, y);
    Float maxxy = max(x, y);

    EXPECT_EQ(absx.value(), 7.F);
    EXPECT_EQ(absy.value(), 5.F);
    EXPECT_EQ(min(x, y).value(), -5.F);
    EXPECT_EQ(max(x, y).value(), 7.F);

    ASSERT_EQ(absx.variables().size(), 1U);
    ASSERT_EQ(absy.variables().size(), 1U);
    ASSERT_EQ(minxy.variables().size(), 2U);
    ASSERT_EQ(maxxy.variables().size(), 2U);

    EXPECT_EQ(min(absx, absy).value(), 5.F);
    EXPECT_EQ(max(-absx, -absy).value(), -5.F);

    auto ex = absx.eval();
    auto ey = absy.eval();
    EXPECT_EQ(ex.value, absx.value());
    EXPECT_EQ(ey.value, absy.value());
    ASSERT_EQ(ex.derivatives.size(), 1);
    ASSERT_EQ(ey.derivatives.size(), 1);

    ASSERT_EQ(ex.derivatives.begin()->second, 1.F);
    ASSERT_EQ(ey.derivatives.begin()->second, -1.F);
}

TEST(AutoDfTest, SinCosTest)
{
    Float x = 7.F;
    Float sinx = sin(x);
    Float cosx = cos(x);

    EXPECT_EQ(sinx.value(), std::sin(7.F));
    EXPECT_EQ(cosx.value(), std::cos(7.F));
    ASSERT_EQ(sinx.variables().size(), 1U);
    ASSERT_EQ(cosx.variables().size(), 1U);

    auto e1 = sinx.eval();
    auto e2 = cosx.eval();

    EXPECT_EQ(e1.value, sinx.value());
    EXPECT_EQ(e2.value, cosx.value());
    ASSERT_EQ(e1.derivatives.size(), 1);
    ASSERT_EQ(e2.derivatives.size(), 1);

    EXPECT_EQ(e1.derivatives.begin()->second, e2.value);
    EXPECT_EQ(e2.derivatives.begin()->second, -e1.value);
}

TEST(AutoDfTest, SimpleGradientDescentTest)
{
    Float x = 0.5F;
    Float formula = -cos(x);
    EXPECT_NEAR(formula.value(), -cos(x.value()), 1E-6F);
    ASSERT_EQ(formula.variables().size(), 1U);

    auto result = GradientDescent(formula, {NAN, 1e-8F});
    ASSERT_EQ(result.derivatives.size(), 1U);

    EXPECT_NEAR(result.value, -1.F, 1e-5);
    EXPECT_NEAR(x.value(), 0.F, 1e-5);
}

TEST(AutoDfTest, GradientDescentTest)
{
    Float x = 0.5F;
    Float formula = min(-cos(x) + 0.5F * abs(x + 2.F) + (x - 1.F) * (x - 1.F) * 0.1F, 5.F);

    Float::Evaluation result;
    ASSERT_NO_THROW(result = GradientDescent(formula, {NAN, 1e-6F}));

    ASSERT_EQ(result.derivatives.size(), 1U);

    EXPECT_NEAR(result.value, 0.062334731, 1e-5);
    EXPECT_NEAR(x.value(), -0.2522214055, 1e-5);
}

TEST(AutoDfTest, Atan2Test)
{
    Float x = 0.F;
    Float y = 1.0F;
    auto val = atan2(y, x);
    EXPECT_NEAR(val(), M_PI_2, 1e-5);
    EXPECT_NEAR(val.eval().derivatives[x.ID()], -1.F, 1e-5);
    EXPECT_NEAR(val.eval().derivatives[y.ID()], 0.F, 1e-5);

    y = 2.0F;
    auto e = val.eval();
    EXPECT_NEAR(val(), M_PI_2, 1e-5);
    EXPECT_NEAR(e.derivatives[x.ID()], -0.5F, 1e-5);
    EXPECT_NEAR(e.derivatives[y.ID()], 0.F, 1e-5);

    y = -1.0F;
    e = val.eval();
    EXPECT_NEAR(val(), -M_PI_2, 1e-5);
    EXPECT_NEAR(e.derivatives[x.ID()], 1.F, 1e-5);
    EXPECT_NEAR(e.derivatives[y.ID()], 0.F, 1e-5);

    x = 1.F;
    y = 1.F;
    e = val.eval();
    EXPECT_NEAR(val(), M_PI_4, 1e-5);
    EXPECT_NEAR(e.derivatives[x.ID()], -0.5, 1e-5);
    EXPECT_NEAR(e.derivatives[y.ID()], 0.5F, 1e-5);

    x = -1.F;
    y = -1.F;
    e = val.eval();
    EXPECT_NEAR(val(), -M_PI_2 - M_PI_4, 1e-5);
    EXPECT_NEAR(e.derivatives[x.ID()], 0.5, 1e-5);
    EXPECT_NEAR(e.derivatives[y.ID()], -0.5F, 1e-5);
}

TEST(AutoDfTest, TestAll)
{
    Float x(1.F, false);
    Float y = max(elu(lrelu(relu(pow2(x)))), x);
    Float z = sqrt(y) + log(y) + log10(y) + exp(x) + elu(x) + lrelu(x);
    Float w = elu(x) + lrelu(x);
    EXPECT_EQ(y(), 1.F);
    EXPECT_EQ(y.eval().value, 1.F);
    EXPECT_GT(z.eval().value, 5.F);
    EXPECT_EQ(w.eval().value, 2.F);
    w += z;
    EXPECT_GT(w.eval().value, 7.F);
}