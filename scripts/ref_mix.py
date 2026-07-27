import math

R_UNIV = 8.31451

# Species data
species = {
    'N2': {
        'M': 28.0134,
        'coeffs': [
            [2.210371497e4, -3.818461820e2, 6.082738360, -8.530914410e-3, 1.384646189e-5, -9.625793620e-9, 2.519705809e-12],
            [5.877124060e5, -2.239249073e3, 6.066949220, -6.139685500e-4, 1.491806679e-7, -1.923105485e-11, 1.061954386e-15],
            [8.310139160e8, -6.420733540e5, 2.020264635e2, -3.065092046e-2, 2.486903333e-6, -9.705954110e-11, 1.437538881e-15]
        ],
        'b1': [7.108460860e2, 1.283210415e4, 4.938707040e6],
        'b2': [-1.076003744e1, -1.586640027e1, -1.672099740e3],
        'T_breaks': [1000.0, 6000.0]
    },
    'O2': {
        'M': 31.9988,
        'coeffs': [
            [-3.425563420e4, 4.847000970e2, 1.119010961, 4.293889240e-3, -6.836300520e-7, -2.023372700e-9, 1.039040018e-12],
            [-1.037939022e6, 2.344830282e3, 1.819732036, 1.267847582e-3, -2.188067988e-7, 2.053719572e-11, -8.193467050e-16],
            [4.975294300e8, -2.866106874e5, 6.690352250e1, -6.169959020e-3, 3.016396027e-7, -7.421416600e-12, 7.278175770e-17]
        ],
        'b1': [-3.391454870e3, -1.689010929e4, 2.293554027e6],
        'b2': [1.849699470e1, 1.738716506e1, -5.530621610e2],
        'T_breaks': [1000.0, 6000.0]
    },
    'NO': {
        'M': 30.0061,
        'coeffs': [
            [-1.143916503e4, 1.536467592e2, 3.431468730, -2.668592368e-3, 8.481399120e-6, -7.685111050e-9, 2.386797655e-12],
            [2.239018716e5, -1.289651623e3, 5.433936030, -3.656034900e-4, 9.880966450e-8, -1.416076856e-11, 9.380184620e-16],
            [-9.575303540e8, 5.912434480e5, -1.384566826e2, 1.694339403e-2, -1.007351096e-6, 2.912584076e-11, -3.295109350e-16]
        ],
        'b1': [9.098214410e3, 1.750317656e4, -4.677501240e6],
        'b2': [6.728725490, -8.501669090, 1.242081216e3],
        'T_breaks': [1000.0, 6000.0]
    },
    'N': {
        'M': 14.0067,
        'coeffs': [
            [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0],
            [8.876501380e4, -1.071231500e2, 2.362188287, 2.916720081e-4, -1.729515100e-7, 4.012657880e-11, -2.677227571e-15],
            [5.475181050e8, -3.107574980e5, 6.916782740e1, -6.847988130e-3, 3.827572400e-7, -1.098367709e-11, 1.277986024e-16]
        ],
        'b1': [5.610463780e4, 5.697351330e4, 2.550585618e6],
        'b2': [4.193905036, 4.865231506, -5.848769753e2],
        'T_breaks': [1000.0, 6000.0]
    },
    'O': {
        'M': 15.9994,
        'coeffs': [
            [-7.953611300e3, 1.607177787e2, 1.966226438, 1.013670310e-3, -1.110415423e-6, 6.517507500e-10, -1.584779251e-13],
            [2.619020262e5, -7.298722030e2, 3.317177270, -4.281334360e-4, 1.036104594e-7, -9.438304330e-12, 2.725038297e-16],
            [1.779004264e8, -1.082328257e5, 2.810778365e1, -2.975232262e-3, 1.854997534e-7, -5.796231540e-12, 7.191720164e-17]
        ],
        'b1': [2.840362437e4, 3.392428060e4, 8.890942630e5],
        'b2': [8.404241820, -6.679585350e-1, -2.181728151e2],
        'T_breaks': [1000.0, 6000.0]
    }
}

def select_interval(T, T_breaks, n_intervals):
    if n_intervals <= 1:
        return 0
    if T <= T_breaks[0]:
        return 0
    if n_intervals == 2:
        return 1
    if T <= T_breaks[1]:
        return 1
    return 2

def eval_cp_R(T, a):
    invT = 1.0 / T
    return a[0]*invT*invT + a[1]*invT + a[2] + a[3]*T + a[4]*T*T + a[5]*T**3 + a[6]*T**4

def eval_h_RT(T, a, b1):
    invT = 1.0 / T
    lnT = math.log(T)
    return -a[0]*invT*invT + a[1]*lnT*invT + a[2] + a[3]*T*0.5 + a[4]*T*T/3.0 + a[5]*T**3*0.25 + a[6]*T**4*0.2 + b1*invT

def eval_s_R(T, a, b2):
    invT = 1.0 / T
    lnT = math.log(T)
    return -a[0]*0.5*invT*invT - a[1]*invT + a[2]*lnT + a[3]*T + a[4]*T*T*0.5 + a[5]*T**3/3.0 + a[6]*T**4*0.25 + b2

def species_cp_R(sp_name, T):
    sp = species[sp_name]
    iv = select_interval(T, sp['T_breaks'], 3)
    return eval_cp_R(T, sp['coeffs'][iv])

def species_h_RT(sp_name, T):
    sp = species[sp_name]
    iv = select_interval(T, sp['T_breaks'], 3)
    return eval_h_RT(T, sp['coeffs'][iv], sp['b1'][iv])

def species_s_R(sp_name, T):
    sp = species[sp_name]
    iv = select_interval(T, sp['T_breaks'], 3)
    return eval_s_R(T, sp['coeffs'][iv], sp['b2'][iv])

def R_specific(sp_name):
    return R_UNIV / species[sp_name]['M'] * 1000

def species_cp(sp_name, T):
    return R_specific(sp_name) * species_cp_R(sp_name, T)

def species_h(sp_name, T):
    return R_specific(sp_name) * T * species_h_RT(sp_name, T)

def species_s(sp_name, T):
    return R_specific(sp_name) * species_s_R(sp_name, T)

# Reference values for individual species at 300K
T = 300.0
for sp_name in ['N2', 'O2', 'NO', 'N', 'O']:
    cp_R = species_cp_R(sp_name, T)
    h_RT = species_h_RT(sp_name, T)
    s_R = species_s_R(sp_name, T)
    Rs = R_specific(sp_name)
    cp = species_cp(sp_name, T)
    h = species_h(sp_name, T)
    s = species_s(sp_name, T)
    print('{} at 300K: R_spec={:.6f}, cp_R={:.12f}, h_RT={:.12f}, s_R={:.12f}, cp={:.8f}, h={:.8f}, s={:.8f}'.format(
        sp_name, Rs, cp_R, h_RT, s_R, cp, h, s))

# Mix at 300K with stoichiometric air fractions
Y_air = {'N2': 0.755, 'O2': 0.231, 'NO': 0.0, 'N': 0.0, 'O': 0.014}
mix_R = sum(Y_air[sp] * R_specific(sp) for sp in Y_air)
mix_cp = sum(Y_air[sp] * species_cp(sp, T) for sp in Y_air)
mix_h = sum(Y_air[sp] * species_h(sp, T) for sp in Y_air)
mix_gamma = mix_cp / (mix_cp - mix_R)
print()
print('Air mix at 300K:')
print('  mix_R  = {:.8f} J/(kg*K)'.format(mix_R))
print('  mix_cp = {:.8f} J/(kg*K)'.format(mix_cp))
print('  mix_h  = {:.8f} J/kg'.format(mix_h))
print('  gamma  = {:.8f}'.format(mix_gamma))

# Single species N2 for T_from_e test
# At T=300K: e = h - R*T
# Already have h and R
print()
sp_name = 'N2'
h_n2 = species_h(sp_name, 300.0)
R_n2 = R_specific(sp_name)
e_n2 = h_n2 - R_n2 * 300.0
print('N2 at 300K:')
print('  h = {:.8f} J/kg'.format(h_n2))
print('  R = {:.8f} J/(kg*K)'.format(R_n2))
print('  e = h - R*T = {:.8f} J/kg'.format(e_n2))

# Also compute at 1500K
T2 = 1500.0
h_n2_1500 = species_h(sp_name, T2)
R_n2_1500 = R_specific(sp_name)
e_n2_1500 = h_n2_1500 - R_n2_1500 * T2
print('N2 at 1500K:')
print('  h = {:.8f} J/kg'.format(h_n2_1500))
print('  e = {:.8f} J/kg'.format(e_n2_1500))

# Equal mixture N2+O2 at 300K
Y_2sp = {'N2': 0.5, 'O2': 0.5}
mix_R_2 = sum(Y_2sp[sp] * R_specific(sp) for sp in Y_2sp)
mix_cp_2 = sum(Y_2sp[sp] * species_cp(sp, T) for sp in Y_2sp)
mix_h_2 = sum(Y_2sp[sp] * species_h(sp, T) for sp in Y_2sp)
mix_gamma_2 = mix_cp_2 / (mix_cp_2 - mix_R_2)
print()
print('50/50 N2+O2 mix at 300K:')
print('  mix_R  = {:.8f} J/(kg*K)'.format(mix_R_2))
print('  mix_cp = {:.8f} J/(kg*K)'.format(mix_cp_2))
print('  mix_h  = {:.8f} J/kg'.format(mix_h_2))
print('  gamma  = {:.8f}'.format(mix_gamma_2))
print('  cp/cv  = {:.8f}'.format(mix_cp_2 / (mix_cp_2 - mix_R_2)))

# N2 at 300K for T_from_e
# Actually for the T_from_e test, let's use T_target and compute e, then verify recovery
T_target = 2500.0
h_tgt = species_h('N2', T_target)
R_n2_tgt = R_specific('N2')
e_tgt = h_tgt - R_n2_tgt * T_target
print()
print('N2 at {}K for T_from_e:'.format(T_target))
print('  h = {:.8f} J/kg'.format(h_tgt))
print('  e = {:.8f} J/kg'.format(e_tgt))
