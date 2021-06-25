## tiny_autodf : Tiny header-only Automatic Differentiation library (C++)

![CMake Test Status](https://github.com/sergehog/tiny_autodf/actions/workflows/master.yml/badge.svg?branch=master)

Attemp to create simple to use AD mechanism, which could be easily integrated into existing formulas and algorithms, making them essentially differentiable.

# Usage Example (drop-in replacement)

Assume situation, where you use a function, templated with a base type
```
float x = some_value;
float y = function<float>(x);

```
In order to convert it into a differentiable version you only need to redefine base type like that:

```
using Float = tyny_autodf::AutoDf<float>;
Float x {some_value};
Float y = function<Float>(x);
```

Now your `y` is differentiable over the `x`.


