"""Float32 reference values for NASA-9 polynomial testing.

Since Real=float (default build), expected values must account
for float-precision coefficient storage.  These values are computed
with numpy float32 to match the C++ evaluation path.
"""
import numpy as np

# N2, interval 0 (200-1000K)
a0 = np.float32([2.210371497e4, -3.818461820e2, 6.082738360,
                 -8.530914410e-3, 1.384646189e-5, -9.625793620e-9,
                  2.519705809e-12])
b1_0 = np.float32(7.108460860e2)
b2_0 = np.float32(-1.076003744e1)

# N2, interval 1 (1000-6000K)
a1 = np.float32([5.877124060e5, -2.239249073e3, 6.066949220,
                 -6.139685500e-4, 1.491806679e-7, -1.923105485e-11,
                  1.061954386e-15])
b1_1 = np.float32(1.283210415e4)
b2_1 = np.float32(-1.586640027e1)

def _cp_R(a, T):
    T = np.float32(T); invT = np.float32(1)/T
    return a[0]*invT*invT + a[1]*invT + a[2] + a[3]*T + a[4]*T*T + a[5]*T**3 + a[6]*T**4

def _h_RT(a, b1, T):
    T = np.float32(T); invT = np.float32(1)/T
    lnT = np.log(T).astype(np.float32)
    h = -a[0]*invT*invT + a[1]*lnT*invT + a[2] \
        + a[3]*T*np.float32(0.5) + a[4]*T*T/np.float32(3) \
        + a[5]*T**3*np.float32(0.25) + a[6]*T**4*np.float32(0.2) \
        + b1*invT
    return h

def _s_R(a, b2, T):
    T = np.float32(T); invT = np.float32(1)/T
    lnT = np.log(T).astype(np.float32)
    return -a[0]*np.float32(0.5)*invT*invT - a[1]*invT \
           + a[2]*lnT + a[3]*T + a[4]*T*T*np.float32(0.5) \
           + a[5]*T**3/np.float32(3) + a[6]*T**4*np.float32(0.25) + b2

# --- Interval 0 at 300K ---
cp_R_300 = _cp_R(a0, 300)
h_RT_300 = _h_RT(a0, b1_0, 300)
s_R_300 = _s_R(a0, b2_0, 300)
print('Interval 0, T=300K:')
print(f'  cp/R = {cp_R_300:.12f}  # {cp_R_300!s}')
print(f'  h/RT = {h_RT_300:.12f}  # {h_RT_300!s}')
print(f'  s/R  = {s_R_300:.12f}  # {s_R_300!s}')

# --- Interval 0 at 1000K ---
cp_R_1000_0 = _cp_R(a0, 1000)
print('\nInterval 0, T=1000K:')
print(f'  cp/R = {cp_R_1000_0:.12f}')

# --- Interval 1 at 1000K ---
cp_R_1000_1 = _cp_R(a1, 1000)
h_RT_1000_1 = _h_RT(a1, b1_1, 1000)
s_R_1000_1 = _s_R(a1, b2_1, 1000)
print('\nInterval 1, T=1000K:')
print(f'  cp/R = {cp_R_1000_1:.12f}')
print(f'  h/RT = {h_RT_1000_1:.12f}')
print(f'  s/R  = {s_R_1000_1:.12f}')

# --- Interval 1 at 1500K ---
cp_R_1500 = _cp_R(a1, 1500)
h_RT_1500 = _h_RT(a1, b1_1, 1500)
s_R_1500 = _s_R(a1, b2_1, 1500)
print('\nInterval 1, T=1500K:')
print(f'  cp/R = {cp_R_1500:.12f}')
print(f'  h/RT = {h_RT_1500:.12f}')
print(f'  s/R  = {s_R_1500:.12f}')

# --- O2, interval 0 at 300K ---
o2_a0 = np.float32([-3.425563420e4, 4.847000970e2, 1.119010961,
                     4.293889240e-3, -6.836300520e-7, -2.023372700e-9,
                     1.039040018e-12])
o2_b1_0 = np.float32(-3.391454870e3)
o2_b2_0 = np.float32(1.849699470e1)

print('\nO2 interval 0, T=300K:')
print(f'  cp/R = {_cp_R(o2_a0, 300):.12f}')
print(f'  h/RT = {_h_RT(o2_a0, o2_b1_0, 300):.12f}')
print(f'  s/R  = {_s_R(o2_a0, o2_b2_0, 300):.12f}')
