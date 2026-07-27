import math

# N2 coefficients, interval 0 (200-1000K)
a = [2.210371497e4, -3.818461820e2, 6.082738360, -8.530914410e-3, 1.384646189e-5, -9.625793620e-9, 2.519705809e-12]
b1 = 7.108460860e2
b2 = -1.076003744e1

R_UNIV = 8.31451
M_N2 = 28.0134

def eval_cp_R(T):
    invT = 1.0 / T
    return a[0]*invT*invT + a[1]*invT + a[2] + a[3]*T + a[4]*T*T + a[5]*T**3 + a[6]*T**4

def eval_h_RT(T):
    invT = 1.0 / T
    lnT = math.log(T)
    return -a[0]*invT*invT + a[1]*lnT*invT + a[2] + a[3]*T*0.5 + a[4]*T*T/3.0 + a[5]*T**3*0.25 + a[6]*T**4*0.2 + b1*invT

def eval_s_R(T):
    invT = 1.0 / T
    lnT = math.log(T)
    return -a[0]*0.5*invT*invT - a[1]*invT + a[2]*lnT + a[3]*T + a[4]*T*T*0.5 + a[5]*T**3/3.0 + a[6]*T**4*0.25 + b2

T300 = 300.0
print('N2 at T=300K (interval 0):')
print('  cp/R  = {:.12f}'.format(eval_cp_R(T300)))
print('  h/RT  = {:.12f}'.format(eval_h_RT(T300)))
print('  s/R   = {:.12f}'.format(eval_s_R(T300)))

# N2 interval 1 (1000-6000K)
a1 = [5.877124060e5, -2.239249073e3, 6.066949220, -6.139685500e-4, 1.491806679e-7, -1.923105485e-11, 1.061954386e-15]
b1_1 = 1.283210415e4
b2_1 = -1.586640027e1

def eval_cp_R1(T):
    invT = 1.0 / T
    return a1[0]*invT*invT + a1[1]*invT + a1[2] + a1[3]*T + a1[4]*T*T + a1[5]*T**3 + a1[6]*T**4

def eval_h_RT1(T):
    invT = 1.0 / T
    lnT = math.log(T)
    return -a1[0]*invT*invT + a1[1]*lnT*invT + a1[2] + a1[3]*T*0.5 + a1[4]*T*T/3.0 + a1[5]*T**3*0.25 + a1[6]*T**4*0.2 + b1_1*invT

def eval_s_R1(T):
    invT = 1.0 / T
    lnT = math.log(T)
    return -a1[0]*0.5*invT*invT - a1[1]*invT + a1[2]*lnT + a1[3]*T + a1[4]*T*T*0.5 + a1[5]*T**3/3.0 + a1[6]*T**4*0.25 + b2_1

T1000 = 1000.0
T1500 = 1500.0
print()
print('N2 at T=1000K (interval boundary):')
print('  cp/R (iv0) = {:.12f}'.format(eval_cp_R(T1000)))
print('  cp/R (iv1) = {:.12f}'.format(eval_cp_R1(T1000)))
print('  h/RT (iv0) = {:.12f}'.format(eval_h_RT(T1000)))
print('  h/RT (iv1) = {:.12f}'.format(eval_h_RT1(T1000)))
print('  s/R  (iv0) = {:.12f}'.format(eval_s_R(T1000)))
print('  s/R  (iv1) = {:.12f}'.format(eval_s_R1(T1000)))

print()
print('N2 at T=1500K (interval 1):')
print('  cp/R  = {:.12f}'.format(eval_cp_R1(T1500)))
print('  h/RT  = {:.12f}'.format(eval_h_RT1(T1500)))
print('  s/R   = {:.12f}'.format(eval_s_R1(T1500)))

# O2 interval 0 (200-1000K)
ao2 = [-3.425563420e4, 4.847000970e2, 1.119010961, 4.293889240e-3, -6.836300520e-7, -2.023372700e-9, 1.039040018e-12]
b1_o2 = -3.391454870e3
b2_o2 = 1.849699470e1

def eval_cp_R_o2(T):
    invT = 1.0 / T
    return ao2[0]*invT*invT + ao2[1]*invT + ao2[2] + ao2[3]*T + ao2[4]*T*T + ao2[5]*T**3 + ao2[6]*T**4

def eval_h_RT_o2(T):
    invT = 1.0 / T
    lnT = math.log(T)
    return -ao2[0]*invT*invT + ao2[1]*lnT*invT + ao2[2] + ao2[3]*T*0.5 + ao2[4]*T*T/3.0 + ao2[5]*T**3*0.25 + ao2[6]*T**4*0.2 + b1_o2*invT

def eval_s_R_o2(T):
    invT = 1.0 / T
    lnT = math.log(T)
    return -ao2[0]*0.5*invT*invT - ao2[1]*invT + ao2[2]*lnT + ao2[3]*T + ao2[4]*T*T*0.5 + ao2[5]*T**3/3.0 + ao2[6]*T**4*0.25 + b2_o2

print()
print('O2 at T=300K (interval 0):')
print('  cp/R  = {:.12f}'.format(eval_cp_R_o2(T300)))
print('  h/RT  = {:.12f}'.format(eval_h_RT_o2(T300)))
print('  s/R   = {:.12f}'.format(eval_s_R_o2(T300)))

# N2 transport, interval 0 (200-1000K)
A, B, C, D = 0.62526577, -31.779652, -1640.7983, 1.7454992
def eval_mu(T):
    ln_mu = A*math.log(T) + B/T + C/(T*T) + D
    return math.exp(ln_mu) * 1e-7

print()
print('N2 transport mu:')
print('  mu(500K) = {:.12e} Pa*s'.format(eval_mu(500.0)))
print('  mu(300K) = {:.12e} Pa*s'.format(eval_mu(300.0)))
print('  mu(1000K)= {:.12e} Pa*s'.format(eval_mu(1000.0)))

# N2 conductivity, interval 0 (200-1000K)
Ak, Bk, Ck, Dk = 0.85439436, 105.73224, -12347.848, 0.47793128
def eval_kappa(T):
    ln_k = Ak*math.log(T) + Bk/T + Ck/(T*T) + Dk
    return math.exp(ln_k) * 1e-7

print()
print('N2 transport kappa:')
print('  kappa(500K) = {:.12e} W/(m*K)'.format(eval_kappa(500.0)))
print('  kappa(300K) = {:.12e} W/(m*K)'.format(eval_kappa(300.0)))
print('  kappa(1000K)= {:.12e} W/(m*K)'.format(eval_kappa(1000.0)))
