# %%
# Import only the necessary libraries for data loading and plotting
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets
from IPython.display import display

# Try to set up the interactive backend robustly for the interactive widget
try:
    from IPython import get_ipython
    ipy = get_ipython()
    if ipy:
        import ipympl
        ipy.run_line_magic('matplotlib', 'widget')
except (ImportError, RuntimeError, Exception):
    print("⚠️ Warning: 'ipympl' is not installed or 'widget' backend is unavailable.")
    print("   Run `pip install ipympl` and restart the kernel to enable clicking/sliders.")
    if ipy:
        ipy.run_line_magic('matplotlib', 'inline')

# %%
# Load configuration to get directory paths
checkpoint_path = '/dat/andreuva/gpu/graphnet/graphnet_nlte/checkpoints/multistride_cpudtst_4x4_s4_m8_b8_r6_d025_press/2026.03.01-02:16:27_best.pth'
checkpoint = torch.load(checkpoint_path, weights_only=False)
config = checkpoint['config']
savedir = config['training']['savedir']

print(f"Successfully loaded configuration from checkpoint {checkpoint_path}.")

# %%
# Define parameters
stride = 3
dataset_type = 'full'
nwave = 1001
wave = np.linspace(853.9444, 854.9444, nwave)

# Load the synthesized intensity profiles
print("Loading synthesized profiles...")
Iwave_lte = np.load(f'{savedir}Iwave_lte_stride_{stride}_{dataset_type}.npy')
Iwave_nlte = np.load(f'{savedir}Iwave_nlte_stride_{stride}_{dataset_type}.npy')
Iwave_target = np.load(f'{savedir}Iwave_target_stride_{stride}_{dataset_type}.npy')
Iwave_predicted = np.load(f'{savedir}Iwave_predicted_stride_{stride}_{dataset_type}.npy')

nx_patch, ny_patch, _ = Iwave_target.shape
pixel_x, pixel_y = nx_patch // 2, ny_patch // 2  # Center pixel for naming/references

print(f"Shape of the synthesized intensities: Predicted: {Iwave_predicted.shape}, Target: {Iwave_target.shape}, LTE: {Iwave_lte.shape}")

# %%
# Calculate Spatially Averaged Spectra
mean_I_lte = np.mean(Iwave_lte, axis=(0, 1))
mean_I_nlte = np.mean(Iwave_nlte, axis=(0,1), where= Iwave_nlte != 0)
mean_I_orig = np.mean(Iwave_target, axis=(0, 1))
mean_I_pred = np.mean(Iwave_predicted, axis=(0, 1))

# Calculate the index for the line core (approx 854.21 nm for Ca II)
line_core_idx = np.argmin(mean_I_orig)

# ==============================================================================
# PLOT 1: INTENSITY MAPS
# ==============================================================================
fig = plt.figure(figsize=(14, 12), dpi=100)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.1, wspace=0.1)

# Determine color scale limits for LTE/NLTE
vmin_lte = min(np.min(Iwave_lte[:, :, line_core_idx]), np.min(Iwave_lte[:, :, line_core_idx])) * 0
vmax_lte = max(np.max(Iwave_lte[:, :, line_core_idx]), np.max(Iwave_lte[:, :, line_core_idx]))

# --- Top Rows: Intensity Maps (LTE/NLTE) ---
ax_map_lte = fig.add_subplot(gs[0, 0])
im_lte = ax_map_lte.imshow(Iwave_lte[:, :, line_core_idx].T, origin='lower', cmap='magma', vmin=vmin_lte, vmax=vmax_lte)
ax_map_lte.set_title('LTE', fontsize=16)
ax_map_lte.set_ylabel('y pixel', fontsize=16)

ax_map_nlte = fig.add_subplot(gs[0, 1], sharey=ax_map_lte)
im_nlte = ax_map_nlte.imshow(Iwave_nlte[:, :, line_core_idx].T, origin='lower', cmap='magma', vmin=vmin_lte, vmax=vmax_lte)
ax_map_nlte.set_title('NLTE 1D', fontsize=16)

cbar1 = fig.colorbar(im_nlte, ax=[ax_map_lte, ax_map_nlte], location='right', pad=0.02, aspect=30)
cbar1.set_label('Intensity [W / (m$^2$ Hz sr)]', fontsize=16)

# Determine color scale limits for Target/Predicted
vmin_pred = min(np.min(Iwave_target[:, :, line_core_idx]), np.min(Iwave_predicted[:, :, line_core_idx]))
vmax_pred = max(np.max(Iwave_target[:, :, line_core_idx]), np.max(Iwave_predicted[:, :, line_core_idx]))

# --- Bottom Rows: Intensity Maps (Target/Predicted) ---
ax_map_target = fig.add_subplot(gs[1, 0], sharex=ax_map_lte)
im_target = ax_map_target.imshow(Iwave_target[:, :, line_core_idx].T, origin='lower', cmap='magma', vmin=vmin_pred, vmax=vmax_pred)
ax_map_target.set_title('Target', fontsize=16)
ax_map_target.set_xlabel('x pixel', fontsize=16)
ax_map_target.set_ylabel('y pixel', fontsize=16)

ax_map_pred = fig.add_subplot(gs[1, 1], sharey=ax_map_target, sharex=ax_map_nlte)
im_pred = ax_map_pred.imshow(Iwave_predicted[:, :, line_core_idx].T, origin='lower', cmap='magma', vmin=vmin_pred, vmax=vmax_pred)
ax_map_pred.set_title('Predicted', fontsize=16)
ax_map_pred.set_xlabel('x pixel', fontsize=16)

cbar2 = fig.colorbar(im_pred, ax=[ax_map_target, ax_map_pred], location='right', pad=0.02, aspect=30)
cbar2.set_label('Intensity [W / (m$^2$ Hz sr)]', fontsize=16)

save_path_dist = os.path.join(savedir, f'synthesis_maps_s{stride}_{dataset_type}_{pixel_x}_{pixel_y}.png')
plt.savefig(save_path_dist, dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# PLOT 2: MEAN SPECTRA
# ==============================================================================
fig = plt.figure(figsize=(12, 4), dpi=100)
plt.plot(wave, mean_I_lte, label='LTE Assumption', color='gray', linestyle='--', alpha=0.7)
plt.plot(wave, mean_I_nlte, label='NLTE 1D', color='blue', linestyle='--', alpha=0.7)
plt.plot(wave, mean_I_orig, label='Target (Non-LTE)', color='black', linewidth=2)
plt.plot(wave, mean_I_pred, label='GNN Prediction', color='red', linestyle=':', linewidth=2)

plt.xlabel('Wavelength [nm]', fontsize=16)
plt.ylabel('Intensity [W / (m$^2$ Hz sr)]', fontsize=16)
plt.legend()
# plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path_dist = os.path.join(savedir, f'synthesis_spectra_s{stride}_{dataset_type}_{pixel_x}_{pixel_y}.png')
plt.savefig(save_path_dist, dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# PLOT 3: INTERACTIVE EXPLORER
# ==============================================================================
vmin_int = min(np.min(Iwave_target), np.min(Iwave_predicted))
vmax_int = max(np.max(Iwave_target), np.max(Iwave_predicted))
sel_x, sel_y = nx_patch // 2, ny_patch // 2

fig = plt.figure(figsize=(12, 6))
gs_int = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])

ax_orig = fig.add_subplot(gs_int[0, 0])
ax_pred = fig.add_subplot(gs_int[0, 1])
ax_spec = fig.add_subplot(gs_int[0, 2])

img_orig = ax_orig.imshow(Iwave_target[:, :, 0].T, origin='lower', cmap='magma', vmin=vmin_int, vmax=vmax_int)
img_pred = ax_pred.imshow(Iwave_predicted[:, :, 0].T, origin='lower', cmap='magma', vmin=vmin_int, vmax=vmax_int)

ax_orig.set_title("Original Intensity")
ax_pred.set_title("Predicted Intensity")
ax_orig.set_xlabel("x pixel")
ax_orig.set_ylabel("y pixel")
ax_pred.set_xlabel("x pixel")

cross_orig, = ax_orig.plot(sel_x, sel_y, 'g+', markersize=15, markeredgewidth=2)
cross_pred, = ax_pred.plot(sel_x, sel_y, 'g+', markersize=15, markeredgewidth=2)

line_lte, = ax_spec.plot(wave, Iwave_lte[sel_x, sel_y, :], 'gray', linestyle='--', alpha=0.6, label='LTE')
line_orig, = ax_spec.plot(wave, Iwave_target[sel_x, sel_y, :], 'k-', linewidth=1.5, label='Target')
line_pred, = ax_spec.plot(wave, Iwave_predicted[sel_x, sel_y, :], 'r:', linewidth=2, label='Prediction')
vline = ax_spec.axvline(wave[0], color='blue', alpha=0.5)

ax_spec.set_title(f"Spectrum at Pixel ({sel_x}, {sel_y})")
ax_spec.set_xlabel("Wavelength [nm]")
ax_spec.set_ylabel("Intensity")
ax_spec.legend(loc='upper right')
# ax_spec.grid(True, alpha=0.3)

def update_maps(wave_idx):
    img_orig.set_data(Iwave_target[:, :, wave_idx].T)
    img_pred.set_data(Iwave_predicted[:, :, wave_idx].T)
    vline.set_xdata([wave[wave_idx], wave[wave_idx]])
    fig.suptitle(f"Wavelength: {wave[wave_idx]:.4f} nm (Index: {wave_idx})", fontsize=14)
    fig.canvas.draw_idle()

def update_spectrum(x, y):
    if x < 5 or x >= nx_patch or y < 5 or y >= ny_patch: return
    line_lte.set_ydata(Iwave_lte[x, y, :])
    line_orig.set_ydata(Iwave_target[x, y, :])
    line_pred.set_ydata(Iwave_predicted[x, y, :])
    cross_orig.set_data([x], [y])
    cross_pred.set_data([x], [y])
    ax_spec.set_title(f"Spectrum at Pixel ({int(x)}, {int(y)})")
    ax_spec.relim()
    ax_spec.autoscale_view()
    fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes in [ax_orig, ax_pred]:
        click_x = int(round(event.xdata))
        click_y = int(round(event.ydata))
        update_spectrum(click_x, click_y)

cid = fig.canvas.mpl_connect('button_press_event', on_click)

slider = widgets.IntSlider(
    value=0, min=0, max=nwave - 1, step=1,
    description='Wave Index:', continuous_update=False,
    layout=widgets.Layout(width='80%')
)
widgets.interactive_output(update_maps, {'wave_idx': slider})

plt.tight_layout()
display(slider)
save_path_dist = os.path.join(savedir, f'comparison_maps_interactive_s{stride}_{dataset_type}.png')
plt.savefig(save_path_dist, dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# PLOT 4: ERROR DISTRIBUTIONS & STATISTICS
# ==============================================================================
epsilon = 1e-32
diff = Iwave_predicted - Iwave_target
rel_error = (diff / (Iwave_target + epsilon)) * 100

median_error_spec = np.median(rel_error, axis=(0, 1))
p05_error_spec = np.percentile(rel_error, 5, axis=(0, 1))
p16_error_spec = np.percentile(rel_error, 16, axis=(0, 1))
p84_error_spec = np.percentile(rel_error, 84, axis=(0, 1))
p95_error_spec = np.percentile(rel_error, 95, axis=(0, 1))
mare_map = np.mean(np.abs(rel_error), axis=2)

fig = plt.figure(figsize=(12, 10))
gs_err = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.2], wspace=0.25, hspace=0.17)

# --- Top: Error Distribution vs Wavelength ---
ax1 = fig.add_subplot(gs_err[0, :])
ax1.fill_between(wave, p05_error_spec, p95_error_spec, color='red', alpha=0.15, label='5th-95th Percentile')
ax1.fill_between(wave, p16_error_spec, p84_error_spec, color='red', alpha=0.4, label='16th-84th Percentile')
ax1.plot(wave, median_error_spec, color='red', linewidth=2, label='Median Error (Bias)')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
# ax1.set_title(r'Relative Error Distribution vs Wavelength', fontsize=16)
ax1.set_ylabel('Relative Error [%]', fontsize=14)
ax1.set_xlabel('Wavelength [nm]', fontsize=14)
ax1.set_xlim(wave.min(), wave.max())
# ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
ax1_twin.plot(wave, mean_I_orig, color='blue', alpha=0.5, linestyle='--', linewidth=1.5, label='Mean Spectrum (Ref)')
ax1_twin.set_ylabel(r'Intensity [W / (m$^2$ Hz sr)]', color='blue', fontsize=14)
ax1_twin.tick_params(axis='y', labelcolor='blue')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right', framealpha=0.9)

# --- Bottom Left: Global Histogram ---
ax2 = fig.add_subplot(gs_err[1, 0])
flat_rel_error = rel_error.flatten()
viz_range = np.percentile(flat_rel_error, [0.1, 99.9])
ax2.hist(flat_rel_error, bins=100, range=viz_range, density=True, color='gray', alpha=1.0)
# ax2.set_title('Global Histogram of Relative Errors', fontsize=14)
ax2.set_xlabel('Relative Error [%]', fontsize=14)
ax2.set_ylabel('Probability Density', fontsize=14)
# ax2.grid(True, alpha=0.3)
ax2.axvline(0, color='black', linestyle='--', linewidth=1)

# --- Bottom Right: Spatial Map of mare ---
ax3 = fig.add_subplot(gs_err[1, 1])
im3 = ax3.imshow(mare_map.T, origin='lower', cmap='inferno', vmin=0, vmax=np.percentile(mare_map, 98), aspect='auto')
# ax3.set_title('Spatial Map of MARE', fontsize=14)
ax3.set_xlabel('x pixel', fontsize=14)
ax3.set_ylabel('y pixel', fontsize=14)
cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label('Mean Absolute Relative Error [%]', fontsize=14)

save_path_dist = os.path.join(savedir, f'synthesis_error_distribution_v2_s{stride}_{dataset_type}.png')
plt.savefig(save_path_dist, dpi=300, bbox_inches='tight')
print(f"Error distribution plot saved to: {save_path_dist}")
plt.show()

# --- Summary Statistics ---
mu, std = np.mean(flat_rel_error), np.std(flat_rel_error)
print("=== Error Statistics ===")
print(f"Global Mean Relative Error (Bias): {mu:.4f} %")
print(f"Global Std Dev of Relative Error:  {std:.4f} %")
print(f"Max Pixel MARE: {np.max(mare_map):.4f} %")
print(f"Mean Pixel MARE: {np.mean(mare_map):.4f} %")