import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import dynamics_learning as dl

sigma = np.array([[[0., 1.],[1., 0.]], [[0., -1j],[1j, 0.]], [[1., 0.],[0., -1.]]])#The set of Pauli matrices


#Initialization of two models and uploading predefined Hamiltonians
model_1 = dl.dynamics_learning(2, 2, 2)
model_2 = dl.dynamics_learning(2, 2, 4)
model_1.set_in_state(np.array([[0., 0.], [0., 1.]]))
model_2.set_in_state(np.array([[0., 0.], [0., 1.]]))
h = np.load('h.npy')
h_true = np.load('h_true.npy')
model_1.set_h(h_true)
model_2.set_h(h)


#Pertubated hamiltonians
h_pert = rnd.randn(2,2)
h_pert = (h_pert + h_pert.conj().T)/2
h_pert1 = 0.5*np.kron(h_pert, np.eye(4,4))
h_pert2 = 0.5*np.kron(h_pert, np.eye(8,8))

#==================================PARAMETERS===================================
plotting_time_step = 0.1
max_total_plotting_time = 200
min_total_plotting_time = 20
#===============================================================================
max_time_steps = int(max_total_plotting_time/plotting_time_step)
min_time_steps = int(min_total_plotting_time/plotting_time_step)

#Dynamics simulation
rho1 = model_1.get_evolution_h_pert_continuous(max_time_steps, plotting_time_step, h_pert1)
rho2 = model_2.get_evolution_h_pert_continuous(max_time_steps, plotting_time_step, h_pert2)
model_1.set_in_state(np.array([[1., 0.], [0., 1.]])/2)
model_2.set_in_state(np.array([[1., 0.], [0., 1.]])/2)
rho_zero1 = model_1.get_evolution_h_pert_continuous(max_time_steps, plotting_time_step, h_pert1)
rho_zero2 = model_2.get_evolution_h_pert_continuous(max_time_steps, plotting_time_step, h_pert2)
model_1.set_in_state(np.array([[0., 0.], [0., 1.]]))
model_2.set_in_state(np.array([[0., 0.], [0., 1.]]))
T = np.arange(0, max_total_plotting_time, plotting_time_step)


#Plotting
sx1 = np.einsum('kij,ij->k', rho1, sigma[0])
sx2 = np.einsum('kij,ij->k', rho2, sigma[0])
sy1 = np.einsum('kij,ij->k', rho1, sigma[1])
sy2 = np.einsum('kij,ij->k', rho2, sigma[1])
sz1 = np.einsum('kij,ij->k', rho1, sigma[2])
sz2 = np.einsum('kij,ij->k', rho2, sigma[2])
fig1 = plt.figure()
fig1.set_size_inches(10, 5)
fig1.subplots_adjust(wspace=0.25, hspace=0.35)
plt.subplot(2,2,1)
plt.ylim(top=1)
plt.ylim(bottom=-1)
plt.plot(T, sx1, 'r')
plt.plot(T, sx2, 'b')
plt.ylabel('$<\sigma_x>$')
plt.subplot(2,2,2)
plt.ylim(top=1)
plt.ylim(bottom=-1)
plt.plot(T, sy1, 'r')
plt.plot(T, sy2, 'b')
plt.ylabel('$<\sigma_y>$')
plt.subplot(2,2,3)
plt.ylim(top=1)
plt.ylim(bottom=-1)
plt.plot(T, sz1, 'r')
plt.plot(T, sz2, 'b')
plt.xlabel('Time')
plt.ylabel('$<\sigma_z>$')
plt.subplot(2,2,4)
diff1 = rho1 - rho_zero1
diff2 = rho2 - rho_zero2
tr_dist1 = np.sqrt(np.einsum('kij,kij->k', diff1, diff1.conj()))
tr_dist2 = np.sqrt(np.einsum('kij,kij->k', diff2, diff2.conj()))
plt.plot(T, tr_dist1, 'r')
plt.plot(T, tr_dist2, 'b')
plt.ylabel('Trace distance')
plt.xlabel('Time')
plt.savefig(f'pic/perturbation_long.pdf')
fig2 = plt.figure()
fig2.set_size_inches(10, 5)
fig2.subplots_adjust(wspace=0.25, hspace=0.35)
plt.subplot(2,2,1)
plt.ylim(top=1)
plt.ylim(bottom=-1)
plt.plot(T[:min_time_steps], sx1[:min_time_steps], 'r')
plt.plot(T[:min_time_steps], sx2[:min_time_steps], 'b')
plt.ylabel('$<\sigma_x>$')
plt.subplot(2,2,2)
plt.ylim(top=1)
plt.ylim(bottom=-1)
plt.plot(T[:min_time_steps], sy1[:min_time_steps], 'r')
plt.plot(T[:min_time_steps], sy2[:min_time_steps], 'b')
plt.ylabel('$<\sigma_y>$')
plt.subplot(2,2,3)
plt.ylim(top=1)
plt.ylim(bottom=-1)
plt.plot(T[:min_time_steps], sz1[:min_time_steps], 'r')
plt.plot(T[:min_time_steps], sz2[:min_time_steps], 'b')
plt.xlabel('Time')
plt.ylabel('$<\sigma_z>$')
plt.subplot(2,2,4)
plt.plot(T[:min_time_steps], tr_dist1[:min_time_steps], 'r')
plt.plot(T[:min_time_steps], tr_dist2[:min_time_steps], 'b')
plt.ylabel('Trace distance')
plt.xlabel('Time')
plt.savefig(f'pic/perturbation.pdf')