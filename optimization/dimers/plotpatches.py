import matplotlib.pyplot as plt

def read_values(file_name):
    with open(file_name, 'r') as file:
        return [list(map(float, line.strip().split(','))) for line in file]

#
grad_values = read_values('grads.txt')
d0_values = read_values('d0s.txt')
yield_values = read_values('yield.txt')

max_length = max(len(grad_values), len(d0_values), len(yield_values))
iterations = range(1, max_length + 1)

fig, axs = plt.subplots(3, 1, figsize=(12, 18))


d0_grads, dr_grads, dg_grads, db_grads = zip(*grad_values)

axs[0].plot(iterations[:len(d0_grads)], d0_grads, 'r', marker='o', label='D0 Gradient')
axs[0].plot(iterations[:len(dr_grads)], dr_grads, 'b', marker='o', label='DR Gradient')
axs[0].plot(iterations[:len(dg_grads)], dg_grads, 'g', marker='o', label='DG Gradient')
axs[0].plot(iterations[:len(db_grads)], db_grads, 'y', marker='o', label='DB Gradient')
axs[0].set_title('Optimization Gradient Values')
axs[0].set_xlabel('Iteration Number')
axs[0].set_ylabel('Gradient Value')
axs[0].legend()


d0s, drs, dgs, dbs = zip(*d0_values)

axs[1].plot(iterations[:len(d0s)], d0s, 'r', marker='x', label='D0 Value')
axs[1].plot(iterations[:len(drs)], drs, 'b', marker='x', label='DR Value')
axs[1].plot(iterations[:len(dgs)], dgs, 'g', marker='x', label='DG Value')
axs[1].plot(iterations[:len(dbs)], dbs, 'y', marker='x', label='DB Value')
axs[1].set_title('Optimization Parameter Values')
axs[1].set_xlabel('Iteration Number')
axs[1].set_ylabel('Parameter Value')
axs[1].legend()


axs[2].plot(iterations[:len(yield_values)], yield_values, 'm', marker='^', label='Yield')
axs[2].set_title('Optimization Yield Values')
axs[2].set_xlabel('Iteration Number')
axs[2].set_ylabel('Yield Value')
axs[2].legend()

plt.tight_layout()
plt.savefig('optimization_patches.png')
plt.show()
