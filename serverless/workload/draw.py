import os
from matplotlib import pyplot as plt

with open('sporadic-load.txt', 'r') as f:
    sporadic_load = f.readline(3000).split()
    sporadic_load = [int(i) for i in sporadic_load]

with open('periodic-load.txt', 'r') as f:
    periodic_load = f.readline(3000).split()
    periodic_load = [int(i) for i in periodic_load]

with open('_bursty-load.txt', 'r') as f:
    bursty_load = f.readline(3000).split()
    bursty_load = [int(i) for i in bursty_load]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)  # subplot(nrows, ncols, index)
plt.plot(sporadic_load, label="Sporadic load", color="blue")
plt.ylabel('Number of Requests')
plt.xlabel('Time Unit')
plt.title('Sporadic Load')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(periodic_load, label="Periodic load", color="green")
plt.ylabel('Number of Requests')
plt.xlabel('Time Unit')
plt.title('Periodic Load')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(bursty_load, label="Bursty load", color="red")
plt.ylabel('Number of Requests')
plt.xlabel('Time Unit')
plt.title('Bursty Load')
plt.grid(True)
plt.legend()

plt.tight_layout()

root_path = os.getenv("STREAMBOX_ROOT")
print(root_path)
plt.savefig(f'{root_path}/resource/workload_visible.png')