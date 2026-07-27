import numpy as np

R = np.float32(8.31451)

# N2
M_n2 = np.float32(28.0134)
Rs_n2 = R / M_n2 * np.float32(1000)
# from ref_nasa9_f32.py output
cp_R_n2 = np.float32(3.502934694290)
h_RT_n2 = np.float32(0.021600961685)
s_R_n2 = np.float32(23.066894531250)

cp_n2 = Rs_n2 * cp_R_n2
h_n2 = Rs_n2 * np.float32(300.0) * h_RT_n2
s_n2 = Rs_n2 * s_R_n2
print('N2 species (f32):')
print(f'  Rs = {Rs_n2}')
print(f'  cp = {cp_n2:.8f}')
print(f'  h  = {h_n2:.8f}')
print(f'  s  = {s_n2:.8f}')

# O2
M_o2 = np.float32(31.9988)
Rs_o2 = R / M_o2 * np.float32(1000)
cp_R_o2 = np.float32(3.534485101700)
h_RT_o2 = np.float32(0.021793365479)
s_R_o2 = np.float32(24.695529937744)

cp_o2 = Rs_o2 * cp_R_o2
h_o2 = Rs_o2 * np.float32(300.0) * h_RT_o2
s_o2 = Rs_o2 * s_R_o2
print('O2 species (f32):')
print(f'  Rs = {Rs_o2}')
print(f'  cp = {cp_o2:.8f}')
print(f'  h  = {h_o2:.8f}')
print(f'  s  = {s_o2:.8f}')
