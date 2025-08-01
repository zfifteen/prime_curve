# Proof and Analysis of Relativistic Doppler Shift Bounds

This repository provides a formal proof and numerical verification for simple algebraic bounds on the relativistic velocity parameter $\\beta$ derived from the fractional Doppler blueshift $\\delta$. The entire proof is contained and executed within the `proof.py` Python script, which uses `sympy` for symbolic manipulation.

The analysis proves that for a source moving directly towards an observer:

1.  **Lower Bound:** For any blueshift ($\\delta \> 0$):
    $$\beta > \frac{\delta}{1 + \delta}$$
2.  **Upper Bound:** For a blueshift between 0 and 1 ($0 \< \\delta \< 1$):
    $$\beta < \frac{\delta}{1 - \delta}$$

-----

## Algebraic Proof

The relativistic Doppler effect for an object moving directly towards an observer relates the fractional blueshift $\\delta$ to the velocity parameter $\\beta = v/c$ by:
$$1 + \delta = \sqrt{\frac{1+\beta}{1-\beta}}$$

Solving for $\\beta$ gives the exact expression:
$$\beta = \frac{(1+\delta)^2 - 1}{(1+\delta)^2 + 1} = \frac{\delta^2 + 2\delta}{\delta^2 + 2\delta + 2}$$

The script validates the bounds by taking the difference between $\\beta$ and each bound and proving the difference is always positive.

* **Proof of Lower Bound:** The difference $\\beta - \\frac{\\delta}{1+\\delta}$ simplifies to:
  $$\frac{\delta^2}{(\delta+1)(\delta^2+2\delta+2)}$$
  This expression is always **positive** for $\\delta \> 0$, confirming the lower bound.

* **Proof of Upper Bound:** The difference $\\frac{\\delta}{1-\\delta} - \\beta$ simplifies to:
  $$\frac{\delta^2(2\delta+3)}{(1-\delta)(\delta^2+2\delta+2)}$$
  This expression is always **positive** for $0 \< \\delta \< 1$, confirming the upper bound.

-----

## Connection to Napier's Inequality

The bounds are conceptually related to Napier's inequality via the relativistic rapidity, $\\psi$. We have the identity $\\psi = \\text{artanh}(\\beta) = \\ln(1+\\delta)$.

Napier's inequality states that for any $z \> 0$:
$$\frac{z}{z+1} < \ln(1+z) < z$$

Substituting $z=\\delta$ and applying the monotonically increasing $\\tanh$ function yields alternative (and tighter) bounds:
$$\tanh\left(\frac{\delta}{1+\delta}\right) < \beta < \tanh(\delta)$$

The numerical analysis confirms:

* The simple algebraic lower bound ($\\frac{\\delta}{1+\\delta}$) is **tighter** than the Napier-derived lower bound.
* The Napier-derived upper bound ($\\tanh(\\delta)$) is **tighter** than the simple algebraic upper bound ($\\frac{\\delta}{1-\\delta}$).

-----

## Verification Script

The `proof.py` script automates the validation process.

### Requirements

The script requires the `sympy` library.

```bash
pip install sympy
```

### Usage

To run the proof and see the verification, execute the script from your terminal:

```bash
python proof.py
```

### Output

The script will print the symbolic simplifications and the results of numerical tests, confirming that all bounds hold.

```
Exact β: ((delta + 1)**2 - 1)/((delta + 1)**2 + 1)

Proof of lower bound (β - δ/(1+δ)) simplifies to: delta**2/(delta**3 + 3*delta**2 + 4*delta + 2)

Proof of upper bound (δ/(1-δ) - β) simplifies to: delta**2*(-2*delta - 3)/(delta**3 + delta**2 - 2)

Napier's Inequality: δ/(δ+1) < ln(1+δ) < δ for δ > 0
Napier-based lower bound: tanh(δ/(1+δ))
Napier-based upper bound: tanh(δ)

Numerical Tests:

δ = 0.01
Exact β = 0.009950
Lower bound = 0.009901
Upper bound = 0.010101
Napier lower = 0.009901 (tighter if > simple lower)
Napier upper = 0.010000 (tighter if < simple upper)

δ = 0.1
Exact β = 0.095023
Lower bound = 0.090909
Upper bound = 0.111111
Napier lower = 0.090659 (tighter if > simple lower)
Napier upper = 0.099668 (tighter if < simple upper)

δ = 0.5
Exact β = 0.384615
Lower bound = 0.333333
Upper bound = 1.000000
Napier lower = 0.321513 (tighter if > simple lower)
Napier upper = 0.462117 (tighter if < simple upper)

δ = 1.0
Exact β = 0.600000
Lower bound = 0.500000
No upper bound for δ >= 1
Napier lower = 0.462117 (tighter if > simple lower)

δ = 10.0
Exact β = 0.983607
Lower bound = 0.909091
No upper bound for δ >= 1
Napier lower = 0.720696 (tighter if > simple lower)

All tests pass: The bounds hold for the tested values, supporting the proof.
```