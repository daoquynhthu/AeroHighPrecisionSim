import math

# N2 transport, interval 0 (200-1000K)
A_n2, B_n2, C_n2, D_n2 = 0.62526577, -31.779652, -1640.7983, 1.7454992
Ak_n2, Bk_n2, Ck_n2, Dk_n2 = 0.85439436, 105.73224, -12347.848, 0.47793128

# O2 transport, interval 0 (200-1000K)
A_o2, B_o2, C_o2, D_o2 = 0.60916180, -52.244847, -599.74009, 2.0410801
Ak_o2, Bk_o2, Ck_o2, Dk_o2 = 0.77229167, 6.8463210, -5893.3377, 1.2210365

def eval_mu(A, B, C, D, T):
    ln_mu = A*math.log(T) + B/T + C/(T*T) + D
    return math.exp(ln_mu) * 1e-7

def eval_kappa(A, B, C, D, T):
    ln_k = A*math.log(T) + B/T + C/(T*T) + D
    return math.exp(ln_k) * 1e-7

T500 = 500.0
mu_n2 = eval_mu(A_n2, B_n2, C_n2, D_n2, T500)
mu_o2 = eval_mu(A_o2, B_o2, C_o2, D_o2, T500)
k_n2 = eval_kappa(Ak_n2, Bk_n2, Ck_n2, Dk_n2, T500)
k_o2 = eval_kappa(Ak_o2, Bk_o2, Ck_o2, Dk_o2, T500)

print('Transport at 500K:')
print('  N2 mu(500K)    = {:.12e} Pa*s'.format(mu_n2))
print('  O2 mu(500K)    = {:.12e} Pa*s'.format(mu_o2))
print('  N2 kappa(500K) = {:.12e} W/(m*K)'.format(k_n2))
print('  O2 kappa(500K) = {:.12e} W/(m*K)'.format(k_o2))

# 50/50 N2+O2 mix
mix_mu = 0.5 * mu_n2 + 0.5 * mu_o2
mix_k = 0.5 * k_n2 + 0.5 * k_o2
print('  50/50 mix mu(500K)    = {:.12e} Pa*s'.format(mix_mu))
print('  50/50 mix kappa(500K) = {:.12e} W/(m*K)'.format(mix_k))

# Verify mu_N2(500K) matches existing test reference (2.60124e-5)
print()
print('mu_N2(500K) should be ~2.60124e-5: {:.12e}'.format(mu_n2))
