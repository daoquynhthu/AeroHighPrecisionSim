import math, sys

def kappa(A,B,C,D,T):
    lnT = math.log(T)
    invT = 1.0/T
    lnk = A*lnT + B*invT + C*invT*invT + D
    return math.exp(lnk) * 1e-7

T = 500.0
k_o2 = kappa(0.77229167, 6.8463210, -5893.3377, 1.2210365, T)
k_n2 = kappa(0.85439436, 105.73224, -12347.848, 0.47793128, T)

print(f"k_N2={k_n2:.12e} k_O2={k_o2:.12e}", file=sys.stderr)

def wilke_phi(mu_i, mu_j, M_i, M_j):
    sq = math.sqrt(mu_i / mu_j)
    fr = math.sqrt(math.sqrt(M_j / M_i))
    num = 1.0 + sq * fr
    den = math.sqrt(8.0 * (1.0 + M_i / M_j))
    return num * num / den

M28, M32 = 28.0134, 31.9988
p1 = wilke_phi(k_n2, k_o2, M28, M32)
p2 = wilke_phi(k_o2, k_n2, M32, M28)
print(f"phi_N2_O2={p1:.12f} phi_O2_N2={p2:.12f}", file=sys.stderr)

# Full Wilke mix
X_n2 = 0.5/M28 / (0.5/M28 + 0.5/M32)
X_o2 = 0.5/M32 / (0.5/M28 + 0.5/M32)

den1 = X_n2 + X_o2 * p1
den2 = X_o2 + X_n2 * p2
result = X_n2 * k_n2 / den1 + X_o2 * k_o2 / den2
print(f"Full Wilke kappa_mix={result:.12e}", file=sys.stderr)
print(f"den1={den1:.12e} den2={den2:.12e}", file=sys.stderr)
