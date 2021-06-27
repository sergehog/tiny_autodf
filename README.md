## tiny_autodf : Tiny, header-only Automatic Differentiation (AD) library (C++)

![CMake Test Status](https://github.com/sergehog/tiny_autodf/actions/workflows/master.yml/badge.svg?branch=master)

Attempt to create simple to use AD mechanism, which could be easily integrated into existing code, formulas and algorithms, essentially making them differentiable.
It also contains basic *Gradient Descent* implementation, essentially turning library into non-linear problem solver.

Originally it was developed as part of another [Tiny PGA](https://github.com/sergehog/tiny_pga) project, however it was extracted as being useful in other projects too.

# Usage Example (drop-in replacement)

Assume you have an equation (or function), depending on one or more variables, for instance something like this:
```
float function(float x, float y)
{
    return 0.5F * x * x + 2.0F * x * y + 4.F * sin(y);
}
...
float result = function(0.1F, 0.2F);
```

You may turn it into a differentiable version by redefining numerical type like that:

```
using Float = tiny_autodf::AutoDf<float>;

template<typename type>
type function(type x, type y)
{
    return 0.5F * x * x + 2.0F * x * y + 4.F * sin(y);
}

Float x = 0.1F;
Float y = 0.2F;
Float result = function<Float>(x, y);
```

Now your `result` is "differentiable" over the `x` and 'y'.
In order to obtain gradient in current point (0.1, 0.2) you'd need to call `eval()` member function:
```
auto eval = result.eval();
std::cout << "dx = " << eval.derivatives[x.ID()] << ", dy = " << eval.derivatives[y.ID()] << std::endl;

```



