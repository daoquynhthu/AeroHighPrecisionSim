import math

def eval_mu(A, B, C, D, T):
    ln_mu = A*math.log(T) + B/T + C/(T*T) + D
    return math.exp(ln_mu) * 1e-7

def herning_zipperer(mu, Y, M):
    s = [math.sqrt(m) for m in M]
    num = sum(Y[i]*mu[i]*s[i] for i in range(len(Y)))
    den = sum(Y[i]*s[i] for i in range(len(Y)))
    return num/den

# N2 transport, interval 0 (200-1000K)
A_n2, B_n2, C_n2, D_n2 = 0.62526577, -31.779652, -1640.7983, 1.7454992
Ak_n2, Bk_n2, Ck_n2, Dk_n2 = 0.85439436, 105.73224, -12347.848, 0.47793128
# O2 transport, interval 0 (200-1000K)
A_o2, B_o2, C_o2, D_o2 = 0.60916180, -52.244847, -599.74009, 2.0410801
Ak_o2, Bk_o2, Ck_o2, Dk_o2 = 0.77229167, 6.8463210, -5893.3377, 1.2210365

T500 = 500.0
mu_n2 = eval_mu(A_n2, B_n2, C_n2, D_n2, T500)
mu_o2 = eval_mu(A_o2, B_o2, C_o2, D_o2, T500)
k_n2 = eval_mu(Ak_n2, Bk_n2, Ck_n2, Dk_n2, T500)
k_o2 = eval_mu(Ak_o2, Bk_o2, Ck_o2, Dk_o2, T500)

print('Transport at 500K:')
print('  N2 mu(500K)    = {:.12e} Pa*s'.format(mu_n2))
print('  O2 mu(500K)    = {:.12e} Pa*s'.format(mu_o2))
print('  N2 kappa(500K) = {:.12e} W/(m*K)'.format(k_n2))
print('  O2 kappa(500K) = {:.12e} W/(m*K)'.format(k_o2))

# Herning-Zipperer 50/50 N2+O2 with real molecular weights
M_n2 = 28.0134
M_o2 = 31.9988
Y = [0.5, 0.5]
mu_mix = herning_zipperer([mu_n2, mu_o2], Y, [M_n2, M_o2])
k_mix = herning_zipperer([k_n2, k_o2], Y, [M_n2, M_o2])
print('  Herning-Zipperer with real M:')
print('  50/50 mix mu(500K)    = {:.12e} Pa*s'.format(mu_mix))
print('  50/50 mix kappa(500K) = {:.12e} W/(m*K)'.format(k_mix))

# Compare with old sqrt(29) placeholder
sqrt29 = math.sqrt(29.0)
print()
print('  With old sqrt(29) placeholder:')
s29 = sqrt29
num_mu_old = 0.5 * mu_n2 * s29 + 0.5 * mu_o2 * s29
den_old = 0.5 * s29 + 0.5 * s29
print('  mix mu(500K) = {:.12e} Pa*s  (same as mass-fraction avg)'.format(num_mu_old/den_old))
