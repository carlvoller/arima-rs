# arima-rs

> ARIMA based Time Series Forecasting for Rust.

This library is my attempt to port Python's pmdarima to Rust. This library is still a Work-in-Progress.

### Goals for this library
1. Achieve as close to one-to-one feature parity with `pmdarima`
2. Mimic as close to one-to-one the function signatures of methods in `pmdarima` (Code from `pmdarima` should look similar if not the same as the Rust here)
3. Expose a drop-in Python replacement API to invoke arima-rs from Python
4. Optimise the performance of commonly used utilities like `ndiffs`, `nsdiffs` that run stationarity and seasonality tests
5. Give my contribution to the future of Data Science in Rust

### Rust package alternatives used

As many data science libraries in Python only exist in Python, alternatives from the Rust Data Science Ecosystem has been used in place of some common Python modules. These include:

1. numpy.ndarray --> rust-ndarray
2. pandas --> pola.rs

On top of these replacements, I've also reimplemented a small handful of utility functions from `sklearn` as there is no current alternative available in Rust.

### Optimisation Techniques

Some optimisation techniques I intend to try are:

1. Using a Basic Linear Algebra Subprogram like OpenBLAS through rust-ndarray (unfortunately pola.rs does not support this)
2. tbc