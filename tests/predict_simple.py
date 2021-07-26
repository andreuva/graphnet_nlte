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
