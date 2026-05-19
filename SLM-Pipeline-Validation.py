import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_hermite as herm

# ── Grid ──────────────────────────────────────────────────────────────────────
pixel_pitch = 12.7e-6
Nx, Ny = 1024, 768
x = (np.arange(Nx) - Nx/2) * pixel_pitch
y = (np.arange(Ny) - Ny/2) * pixel_pitch
X, Y = np.meshgrid(x, y)

# ── Hermite-Gaussian mode ─────────────────────────────────────────────────────
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

# ── Input beam ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)
initial_phase = rng.uniform(0, 2 * np.pi, (Ny, Nx))

w0_beam = 2e-3
laser_amplitude = np.exp(-(X**2 + Y**2) / w0_beam**2)
aperture = (np.abs(X) <= Nx*pixel_pitch/2) & (np.abs(Y) <= Ny*pixel_pitch/2)
laser_amplitude *= aperture
SLM_field = laser_amplitude * np.exp(1j * initial_phase)

# ── Target ────────────────────────────────────────────────────────────────────
target_amplitude = np.abs(hg_mode(3, 3, X, Y, w0=2e-3))
target_amplitude /= target_amplitude.max()
target_mask = target_amplitude > 0.5

# ── GS iterations ─────────────────────────────────────────────────────────────
efficiency_history = []

for i in range(150):
    image_field = np.fft.fftshift(np.fft.fft2(SLM_field))
    image_phase = np.angle(image_field)
    image_field = target_amplitude * np.exp(1j * image_phase)
    SLM_field = np.fft.ifft2(np.fft.ifftshift(image_field))
    slm_phase = np.angle(SLM_field)
    slm_phase_q = np.round(slm_phase / (2*np.pi) * 255) / 255 * (2*np.pi)
    SLM_field = laser_amplitude * np.exp(1j * slm_phase_q)

    recon = np.abs(np.fft.fftshift(np.fft.fft2(SLM_field)))**2
    efficiency_history.append(recon[target_mask].sum() / recon.sum())

# ── Focal plane coordinates ───────────────────────────────────────────────────
f_lens = 0.15
lam = 532e-9
dx_focal = lam * f_lens / (Nx * pixel_pitch)
dy_focal = lam * f_lens / (Ny * pixel_pitch)

x_focal = (np.arange(Nx) - Nx/2) * dx_focal
y_focal = (np.arange(Ny) - Ny/2) * dy_focal

recon_raw = np.abs(np.fft.fftshift(np.fft.fft2(SLM_field)))**2

# Fill factor (~90%) — sinc² envelope in focal plane
fill_factor = 0.90
active_fraction = np.sqrt(fill_factor)
 
fx = np.fft.fftshift(np.fft.fftfreq(Nx))
fy = np.fft.fftshift(np.fft.fftfreq(Ny))
FX, FY = np.meshgrid(fx, fy)
 
fill_envelope = np.sinc(active_fraction * FX) ** 2 * \
                np.sinc(active_fraction * FY) ** 2
recon_ff = recon_raw * fill_envelope
 
# Maximum deflection angle — Nyquist limit of the SLM
theta_max = lam / (2 * pixel_pitch)
y_max = f_lens * theta_max
x_max = f_lens * theta_max
 
X_focal, Y_focal = np.meshgrid(x_focal, y_focal)
alias_mask = (np.abs(X_focal) <= x_max) & (np.abs(Y_focal) <= y_max)
target_in_window = (target_amplitude**2)[alias_mask].sum() / \
                   (target_amplitude**2).sum()

# ── Results ───────────────────────────────────────────────────────────────────
efficiency_raw = recon_raw[target_mask].sum() / recon_raw.sum()
efficiency_ff  = recon_ff[target_mask].sum()  / recon_ff.sum()
 
recon_norm = recon_raw / recon_raw.sum()
target_norm = (target_amplitude**2) / (target_amplitude**2).sum()
overlap = (recon_norm * target_norm).sum() / (
    np.sqrt((recon_norm**2).sum()) * np.sqrt((target_norm**2).sum())
)
 
print(f"Mode overlap : {overlap*100:.1f}%  (gate: >90%)")
print(f"Efficiency   : {efficiency_raw*100:.1f}%  (gate: >60%)")
print(f"Gate 0       : {'PASS' if efficiency_raw > 0.6 and overlap > 0.9 else 'FAIL'}")
print(f"\nFill factor penalty    : {(efficiency_raw - efficiency_ff)*100:.2f} pp")
print(f"theta_max              : {np.degrees(theta_max):.3f} deg  ({theta_max*1e3:.2f} mrad)")
print(f"Max spot displacement  : ±{y_max*1e6:.1f} µm")
print(f"Target energy in window: {target_in_window*100:.1f}%")
print(f"Aliasing check         : {'OK' if target_in_window > 0.95 else 'WARNING — target beyond alias limit'}")

# ── Task 4: Beam Expander Ray Trace ───────────────────────────────────────────
f1, f2 = -10e-3, 75e-3
D_in = 1.5e-3
M = f2 / abs(f1)                        # expansion ratio
D_out = M * D_in                        # output beam diameter
separation = f2 + f1                    # correct separation for collimation

panel_h = Ny * pixel_pitch              # 9.75 mm
panel_w = Nx * pixel_pitch              # 13.0 mm

aperture_L1 = 12.7e-3                   # 0.5 inch lens
aperture_L2 = 25.4e-3                   # 1 inch lens

print(f"Expansion ratio     : {M:.1f}x")
print(f"Output beam diam    : {D_out*1e3:.2f} mm")
print(f"Element separation  : {separation*1e3:.0f} mm")
print(f"Panel active area   : {panel_w*1e3:.1f} x {panel_h*1e3:.1f} mm")
print(f"L1 vignetting       : {'OK' if D_in  < aperture_L1 else 'FAIL'}")
print(f"L2 vignetting       : {'OK' if D_out < aperture_L2 else 'FAIL'}")
print(f"Panel fill (height) : {D_out/panel_h*100:.1f}%  (gate: >80%)")
print(f"Gate 0 (expander)   : {'PASS' if D_out/panel_h > 0.8 else 'FAIL'}")
 
# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Phase 0 — GS Hologram Simulation  |  Task 3: LCoS Pixel Model", fontsize=13)
 
im0 = axes[0, 0].pcolormesh(x_focal*1e6, y_focal*1e6, recon_raw, shading='gouraud', cmap='magma')
axes[0, 0].set_title('Reconstructed Intensity (raw)')
axes[0, 0].set_xlabel('x (µm)')
axes[0, 0].set_ylabel('y (µm)')
fig.colorbar(im0, ax=axes[0, 0])
 
im1 = axes[0, 1].pcolormesh(x_focal*1e6, y_focal*1e6, recon_ff, shading='gouraud', cmap='magma')
axes[0, 1].set_title(f'Reconstructed Intensity (fill-factor corrected, FF={fill_factor:.0%})')
axes[0, 1].set_xlabel('x (µm)')
axes[0, 1].set_ylabel('y (µm)')
fig.colorbar(im1, ax=axes[0, 1])
 
cx, cy = Nx//2, Ny//2
crop = 100
im2 = axes[0, 2].pcolormesh(
    x_focal[cx-crop:cx+crop]*1e6,
    y_focal[cy-crop:cy+crop]*1e6,
    fill_envelope[cy-crop:cy+crop, cx-crop:cx+crop],
    shading='gouraud', cmap='viridis')
axes[0, 2].set_title('Sinc² Fill-Factor Envelope (centre crop)')
axes[0, 2].set_xlabel('x (µm)')
axes[0, 2].set_ylabel('y (µm)')
fig.colorbar(im2, ax=axes[0, 2])
 
axes[1, 0].plot(efficiency_history, color='steelblue', lw=1.5, label='Efficiency')
axes[1, 0].axhline(0.6, color='r', linestyle='--', label='Gate threshold (60%)')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Efficiency')
axes[1, 0].set_title('GS Convergence')
axes[1, 0].legend()
 
axes[1, 1].pcolormesh(x_focal*1e6, y_focal*1e6, recon_ff, shading='gouraud', cmap='magma')
rect = plt.Rectangle((-x_max*1e6, -y_max*1e6), 2*x_max*1e6, 2*y_max*1e6,
                      linewidth=1.5, edgecolor='cyan', facecolor='none',
                      linestyle='--', label=f'Alias limit ±{y_max*1e6:.0f} µm')
axes[1, 1].add_patch(rect)
axes[1, 1].set_title('Alias Boundary on Focal Plane')
axes[1, 1].set_xlabel('x (µm)')
axes[1, 1].set_ylabel('y (µm)')
axes[1, 1].legend(fontsize=8)
 
slm_phase_display = np.angle(SLM_field)
axes[1, 2].imshow(slm_phase_display, cmap='hsv', origin='lower',
                   extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3])
axes[1, 2].set_title('SLM Phase Pattern (8-bit quantised)')
axes[1, 2].set_xlabel('x (mm)')
axes[1, 2].set_ylabel('y (mm)')


plt.tight_layout()
plt.show()