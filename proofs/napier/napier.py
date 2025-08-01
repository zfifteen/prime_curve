import sympy as sp
import math

# This Python script provides a complete scientific proof of the bounds on the relativistic velocity parameter β
# derived from the fractional Doppler blueshift δ, as discussed in the X post.
# The bounds are: for δ > 0, β > δ / (1 + δ)
# And for 0 < δ < 1, β < δ / (1 - δ)
# We prove these algebraically using SymPy for symbolic manipulation.
# We also discuss the connection to Napier's inequality and provide numerical tests to verify the bounds
# and attempt to falsify them by checking if they hold for various δ values.
# If any test fails, it would falsify the claim; otherwise, it supports the proof.

# Part 1: Symbolic Proof of the Bounds

# Define the symbol for δ (assuming δ > 0)
delta = sp.symbols('delta', positive=True)

# The exact expression for β from the relativistic Doppler blueshift formula:
# 1 + δ = sqrt((1 + β) / (1 - β))
# Solving for β: β = ((1 + δ)^2 - 1) / ((1 + δ)^2 + 1)
gamma = 1 + delta
beta_exact = (gamma**2 - 1) / (gamma**2 + 1)

print("Exact β:", beta_exact.simplify())

# Lower bound: l = δ / (1 + δ)
lower = delta / (1 + delta)

# Compute β - lower and simplify to show it's positive for δ > 0
diff_lower = beta_exact - lower
simplified_diff_lower = sp.simplify(diff_lower)
print("\nProof of lower bound (β - δ/(1+δ)) simplifies to:", simplified_diff_lower)
# This simplifies to δ² / ((δ + 1) (δ² + 2δ + 2)), which is positive for δ > 0.
# Hence, β > δ / (1 + δ) is proven.

# Upper bound: u = δ / (1 - δ) (only for δ < 1)
upper = delta / (1 - delta)

# Compute upper - β and simplify (assuming 0 < δ < 1)
diff_upper = upper - beta_exact
simplified_diff_upper = sp.simplify(diff_upper)
print("\nProof of upper bound (δ/(1-δ) - β) simplifies to:", simplified_diff_upper)
# This simplifies to (2δ³ + 3δ²) / ((δ - 1) (δ² + 2δ + 2))
# Note: Numerator > 0, denominator < 0 for 0 < δ < 1 (since δ - 1 < 0 and δ² + 2δ + 2 > 0),
# so overall negative? Wait, no: wait, upper - β > 0 means β < upper.
# Actually, since denominator has (δ - 1) which is negative, and the whole is positive because negative denominator times positive numerator? Wait.
# Let's evaluate the sign: numerator 2δ³ + 3δ² = δ² (2δ + 3) > 0
# Denominator: (δ - 1) < 0, (δ² + 2δ + 2) > 0, so denominator < 0
# Positive / negative = negative, but wait, that would mean upper - β < 0, which is wrong.
# Wait, mistake in sign.
# Actually, sp.simplify may output - (something positive) / (1 - δ) ..., but let's adjust.
# To confirm, we can assume δ < 1 and show diff_upper > 0.
# Rewrite upper - beta = [δ (gamma² + 1) - (1 - δ) (gamma² - 1)] / [(1 - δ) (gamma² + 1)]
# But since we know from numerical it holds, and algebraic calculation earlier showed numerator 3δ² + 2δ³ >0, but denominator (1-δ) wait no.
# Wait, common denominator is (1-δ)(gamma² +1)
# But gamma =1+δ, gamma² +1 = (1+δ)^2 +1 =1+2δ+δ² +1=2+2δ+δ²
# upper = δ/(1-δ)
# beta = (gamma² -1)/(gamma² +1)
# So upper - beta = δ/(1-δ) - (gamma² -1)/(gamma² +1)
# To combine: [ δ (gamma² +1) - (1-δ) (gamma² -1) ] / [ (1-δ) (gamma² +1) ]
# Compute numerator: δ (gamma² +1) - (1-δ) (gamma² -1)
# gamma² = (1+δ)^2 =1 +2δ +δ²
# gamma² +1 =2 +2δ +δ²
# gamma² -1 =2δ +δ²
# So δ (2 +2δ +δ²) - (1-δ) (2δ +δ²)
# = 2δ +2δ² +δ³ - (1-δ)(2δ +δ²)
# Expand (1-δ)(2δ +δ²) = 1*(2δ +δ²) - δ (2δ +δ²) =2δ +δ² -2δ² -δ³ =2δ -δ² -δ³
# So numerator = 2δ +2δ² +δ³ - (2δ -δ² -δ³) = 2δ +2δ² +δ³ -2δ +δ² +δ³ = 3δ² +2δ³
# Yes, 3δ² + 2δ³ >0
# Denominator (1-δ) (gamma² +1) = (1-δ)(2+2δ+δ²)
# Since δ<1, 1-δ>0, and 2+2δ+δ²>0, denominator >0
# Numerator >0, so upper - beta >0, yes β < upper.
# Note: in sympy, since 1-δ is written as - (δ -1), it may show as (3δ² +2δ³) / [ -(δ-1) (δ²+2δ+2) ], and since δ-1<0, but it's equivalent to positive.
# SymPy simplify should show it's positive under assumption.

# Part 2: Connection to Napier's Inequality
# Napier's inequality states: for z > 0, z / (z + 1) < ln(1 + z) < z
print("\nNapier's Inequality: δ/(δ+1) < ln(1+δ) < δ for δ > 0")

# In the relativistic context, artanh(β) = ln(1 + δ)
# So y = ln(1 + δ), β = tanh(y)
# Using Napier's, δ/(1+δ) < y < δ
# Since tanh is increasing and concave down, we get:
# tanh(δ/(1+δ)) < β < tanh(δ)
# These are alternative bounds derived directly from Napier's inequality.
# Note that tanh(δ/(1+δ)) > δ/(1+δ + (δ/(1+δ))^2 / 3 + ...) but actually tanh(w) < w, so this lower is stricter than δ/(1+δ) wait no:
# Since tanh(w) < w, tanh(δ/(1+δ)) < δ/(1+δ), so β > tanh(δ/(1+δ)) is a tighter (higher) lower bound than β > δ/(1+δ).
# But the post uses the looser, simpler bound δ/(1+δ), which is easier for manual calculation.
# Similarly, tanh(δ) < δ, so upper tanh(δ) is tighter than δ/(1-δ) (since δ/(1-δ) > δ for δ<0.5, but anyway).
# The algebraic bounds in the post are simpler rational functions, while Napier's gives hyperbolic ones.
# The post likely highlights Napier's as inspiration for bounding via log properties, given the logarithmic compression analogy in relativity.

# Symbolic example of Napier bounds
y = sp.log(1 + delta)
beta = sp.tanh(y)  # But since exact, same as beta_exact
lower_napier = sp.tanh(delta / (1 + delta))
upper_napier = sp.tanh(delta)

print("Napier-based lower bound: tanh(δ/(1+δ))")
print("Napier-based upper bound: tanh(δ)")

# Part 3: Numerical Tests to Verify and Attempt to Falsify
# We test with various δ > 0. If any bound fails, print "Falsified".
# This attempts to falsify by empirical checking across a range of values.

delta_values = [0.01, 0.1, 0.2, 0.5, 0.9, 1.0, 1.5, 10.0, 100.0]  # Include δ=1 and large to test limits

print("\nNumerical Tests:")
falsified = False
for d in delta_values:
    # Compute exact β
    g = 1 + d
    b_exact = (g**2 - 1) / (g**2 + 1)

    # Lower bound
    l = d / (1 + d)

    # Upper bound only if d < 1
    u = d / (1 - d) if d < 1 else None

    print(f"\nδ = {d}")
    print(f"Exact β = {b_exact:.6f}")
    print(f"Lower bound = {l:.6f}")
    if b_exact <= l:
        print("Lower bound failed: β <= lower")
        falsified = True

    if d < 1:
        print(f"Upper bound = {u:.6f}")
        if b_exact >= u:
            print("Upper bound failed: β >= upper")
            falsified = True
    else:
        print("No upper bound for δ >= 1")

    # Also compute Napier bounds for comparison
    l_napier = math.tanh(d / (1 + d))
    u_napier = math.tanh(d) if d < math.inf else 1.0  # tanh large =1

    print(f"Napier lower = {l_napier:.6f} (tighter if > simple lower)")
    if d < 1:
        print(f"Napier upper = {u_napier:.6f} (tighter if < simple upper)")

if falsified:
    print("\nThe bounds are falsified by at least one test.")
else:
    print("\nAll tests pass: The bounds hold for the tested values, supporting the proof.")