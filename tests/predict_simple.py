"""
STALE -- kept for reference only, does not run against the current Si I pipeline.

These scripts date from the CaII/H work: they use the Ca-active atom set, a ColumnMass depth
scale, and Formal.predict() paths that the Si I database no longer produces (test_normalization
also references an undefined `cmass`). For the current verification path see test_prediction.py
and evaluate_intensity.py in the repository root.
"""
from Formal import Formal as graphnet
import numpy as np
import lightweaver as lw
import matplotlib.pyplot as plt

# Load atmosphere
_, atmosRef = lw.multi.read_multi_atmos('../data/models_atmos/FALC_82.atmos')
tau, vturb, vlos, ne,  tt = [atmosRef.tauRef], [atmosRef.vturb], [atmosRef.vlos], [atmosRef.ne], [atmosRef.temperature]

# Load model and predict
model = graphnet()
prediction = model.predict(tau=tau, vturb=vturb, vlos=vlos, TT=tt, ne=ne, readir='../checkpoints/crd/')

# Plot the results
plt.figure(figsize=(15, 10), dpi=200)
for i in range(len(prediction[0][0, :])):
    plt.plot(prediction[0][:, i], color=f'C{i}')

plt.xlabel('node')
plt.ylabel('log_10(n/n*)')
plt.title('departure coefficients computed and predicted')
plt.show()
