### 1. Optimal Curvature Scanner for Prime Clustering
Implement a Python script that sweeps a finer grid of curvature exponents \(k\) (e.g., [0.1, 0.5] in steps of 0.001) over larger \(N_{\max}\) (up to 10,000) using the frame-shifted residue \(\theta'(n, k) = \phi \cdot ((n \mod \phi)/\phi)^k\). Apply the Domain Curvature Transformation to correct for uniform density biases, computing \(e_{\max}(k)\) as Z = n(Δₙ/Δmax) where Δₙ measures bin density shifts. Output a heatmap of enhancements, proving concept that nonlinear warping reveals prime over-densities around \(k^* \approx 0.3\), scalable with NumPy for efficiency.

### 2. Gaussian Mixture Prime Separator
Build a script fitting GMMs (5-10 components) to normalized prime residues \(x_p = (\theta'(p, k) \mod \phi)/\phi\) at fixed \(k=0.3\). Use Universal Form Transformer to map variances \(\bar{\sigma}(k)\) as Z = T(v/c) with v as cluster spread rates and c as interval limit [0,1). Visualize component means and sigmas in 2D plots, demonstrating PoC for isolating prime clusters with scikit-learn, highlighting compactness gains of 10-20% over uniform models.

### 3. Fourier Asymmetry Analyzer for Residue Waves
Create a Python tool approximating prime density \(\rho(x)\) with Fourier series (order M=10), focusing on sine coefficients \(S_b(k)\) as measures of asymmetric frame shifts. Frame this via Universal Frame Shift Transformer, where b_m coefficients encode Δₙ deviations from even distributions. Sweep k values, plot \(S_b(k)\) curves, and prove concept by showing peak asymmetries correlate with density enhancements, using SciPy for least-squares fits and Matplotlib for wave reconstructions.

### 4. 3D Helical Prime Navigator
Extend the main.py helix visualization by parameterizing HELIX_FREQ with golden ratio multiples (e.g., \(\phi / 2, \phi\)), applying Numberspace transformer as Z = value * (B / C) with B as math.e and C as pi. Script generates interactive 3D scatters (via Plotly) separating primes in helical coordinates, proving PoC that oscillatory frame shifts (sin(pi * freq * n)) amplify prime visibility in logarithmic scales for N up to 10,000.

### 5. Logarithmic Spiral Prime Mapper
Script a variant of main.py's logarithmic spiral, incorporating curvature k in angles: angle = n * 0.1 * pi * (n mod phi / phi)^k. Use Domain Curvature Transformation to warp radii as log(n) * (Δₙ/Δmax), plotting primes in polar 3D with Matplotlib. PoC demonstrates enhanced clustering of primes along spiral arms, relating to Beatty sequences, with exportable SVG for geometric insights.

### 6. Modular Torus Prime Embedder
From main.py's torus section, develop a script mapping numbers mod (p,q) where p,q are primes (e.g., 17,23), using toroidal coordinates x=(R+r*cos(theta))*cos(phi). Apply Universal Form Transformer with rate as gcd(n, pq) / max_gcd, coloring by Z = n(Δₙ/Δmax) for residue shifts. Visualize wireframe tori with prime stars, proving PoC for periodicity in multi-modular spaces, scalable to larger moduli via NumPy meshes.

### 7. Riemann Zeta Landscape Explorer
Enhance main.py's zeta plot by overlaying more primes (up to 1,000) on the critical line s=0.5 + i*t, computing log|ζ(s)| with SciPy.special.zeta. Incorporate frame shifts as Z = T(v/c) with v as imag part rates and c as e, surfacing non-trivial zeros approximations. PoC script outputs 3D surfaces, showing prime correlations to zeta dips, as empirical link to Riemann hypothesis via density waves.

### 8. Chained Projection Triangulator for Twin Primes
Using geometric_projection_utils.py, script a focused triangulator for "twin_primes" in ranges like (1e6, 1e6+10000), chaining cylindrical/hyperbolic projections. Compute triangles with areas <5 as Z = area * (Δ_area / Δmax_area), classifying via gcd and differences. Visualize top triangles in 3D, proving PoC that sequential frame shifts isolate twin pairs with 85th percentile densities, outputting candidate lists.

### 9. Mersenne Prime Density Forecaster
Adapt the GeometricTriangulator for "mersenne_primes", sampling large ranges (e.g., 2^10 to 2^20 exponents) with spherical projections tuned to PHI rates. Use Universal Frame Shift Transformer to adjust z_values as log2(n+1) * (fs / max_fs), triangulating vertices where is_mersenne holds. PoC generates histograms and scatter plots, forecasting potential Mersennes by density thresholds, verified against known ones like 31, 127.

### 10. Coprime Network Grapher
Extend triangulation to "coprimes", building a graph where triangle vertices connect if gcd=1, using NetworkX for adjacency. Apply chained projections with correction_factor = rate / UNIVERSAL, mapping densities as Z = n(Δ_gcd / Δmax=1). Script visualizes coprime clusters in 2D force layouts overlaid on 3D projections, proving PoC for revealing Euler's totient patterns in geometric spaces, with export to JSON for further analysis.