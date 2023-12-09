import numpy as np
from matplotlib import pyplot as plt

fig, axs = plt.subplots(figsize=(1920.0 / 300, 1080.0 / 300), ncols = 1, nrows=3)

travel = np.arange(0, 100.0, 0.1)

for i in range(3):
    axs[i].set_xlabel('Travel [m]')

top = axs[0]
speed = 60 + 5 * np.sin(travel / 5.0)

top.set_ylabel('Speed (mph)')
top.set_ylim(bottom=0, top=1.1 * np.max(speed))
top.plot(travel, speed)

mid = axs[1]
elapsed_time = travel + np.sin(travel)

mid.set_ylabel('Elapsed Time (sec)')
mid.set_ylim(bottom=0, top=1.1 * np.max(elapsed_time))
mid.plot(travel, elapsed_time)

bottom = axs[2]

rpm = 4 + 2 * np.sin(travel / 2.0)

bottom.set_ylabel('Rpm (rev / min)')
bottom.set_ylim(bottom=0, top=1.1 * np.max(rpm))
bottom.plot(travel, rpm)

plt.tight_layout()
# plt.show()
plt.savefig('telemetry.png', dpi=300, format='png')
