import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_hermite as herm

#Grid
N = 512
x = np.linspace(-2e-6, 2e-6, N)
y = np.linspace(-2e-6, 2e-6, N)
X, Y = np.meshgrid(x, y)

#Hermite-Gaussian mode
def hg_mode(n, m, x, y, z=0, w0=1e-6, lam=500e-9):
    z_R = np.pi * w0**2 / lam
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)

    if z == 0:
        phase = 1.0
    else:
        R_z = z * (1 + (z_R / z)**2)
        gouy = (n + m + 1) * np.arctan(z / z_R)
        k = 2 * np.pi / lam
        phase = np.exp(1j * (k*z + k*(x**2 + y**2) / (2*R_z) - gouy))

    mode = (w0 / w_z) * \
           herm(n, np.sqrt(2) * x / w_z) * \
           herm(m, np.sqrt(2) * y / w_z) * \
           np.exp(-(x**2 + y**2) / w_z**2)

    return mode * phase

#input beam 
rng = np.random.default_rng(seed=42)
initial_phase = rng.uniform(0, 2 * np.pi, (N, N))

w0_beam = 0.8e-6
laser_amplitude = np.exp(-(X**2 + Y**2) / w0_beam**2)
SLM_field = laser_amplitude * np.exp(1j * initial_phase)


target_amplitude = np.abs(hg_mode(3, 3, X, Y))
target_amplitude /= target_amplitude.max()
target_mask = target_amplitude > 0.5

#GS iteration
efficiency_history = []

for i in range(150):
    image_field = np.fft.fftshift(np.fft.fft2(SLM_field))
    image_phase = np.angle(image_field)
    image_field = target_amplitude * np.exp(1j * image_phase)
    SLM_field = np.fft.ifft2(np.fft.ifftshift(image_field))
    slm_phase = np.angle(SLM_field)
    SLM_field = laser_amplitude * np.exp(1j * slm_phase)

    recon = np.abs(np.fft.fftshift(np.fft.fft2(SLM_field)))**2
    efficiency_history.append(recon[target_mask].sum() / recon.sum())

#Results 
recon = np.abs(np.fft.fftshift(np.fft.fft2(SLM_field)))**2

efficiency = recon[target_mask].sum() / recon.sum()

recon_norm = recon / recon.sum()
target_norm = (target_amplitude**2) / (target_amplitude**2).sum()
overlap = (recon_norm * target_norm).sum() / (
    np.sqrt((recon_norm**2).sum()) * np.sqrt((target_norm**2).sum())
)

print(f"Mode overlap : {overlap*100:.1f}%  (gate: >90%)")
print(f"Efficiency   : {efficiency*100:.1f}%  (gate: >60%)")
print(f"Gate 0       : {'PASS' if efficiency > 0.6 and overlap > 0.9 else 'FAIL'}")

#Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].pcolormesh(X, Y, recon, shading='gouraud', cmap='magma')
axes[0].set_title('Reconstructed Intensity at Image Plane')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')

axes[1].plot(efficiency_history)
axes[1].axhline(0.6, color='r', linestyle='--', label='Gate threshold (60%)')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Efficiency')
axes[1].set_title('GS Convergence')
axes[1].legend()

plt.tight_layout()
plt.show()