import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import dynamics_learning as dl
from dynamics_learning import dynamics_learning as dl1
from scipy.optimize import fmin_l_bfgs_b as bfgs
import os

sigma = np.array([[[0., 1.],[1., 0.]], [[0., -1j],[1j, 0.]], [[1., 0.],[0., -1.]]]) #The set of Pauli matrices

#Initialization of two models (model_1 for dataset generation and model_2 for learning)
#MODEL1 -> DATASET GENERATION

def aux_h_rand(dimention):     #SETIING RANDOM HAMILTONIANS#
    h = rnd.rand(dimention,dimention) + 1j*rnd.rand(dimention,dimention)
    h = h + h.conj().T
    return h

#MODEL -> SET OF DIMENTIONS
sys_dim_model1=2
mem_dim_model1=2
res_dim_model1=2

model_1 = dl.dynamics_learning(sys_dim_model1, mem_dim_model1, res_dim_model1)
model_1.set_in_state(np.array([[0., 0.], [0., 1.]]))

#MODEL2 -> SET OF DIMENTIONS
sys_dim_model2=2
mem_dim_model2=2
res_dim_model2=2

model_2 = dl.dynamics_learning(sys_dim_model2, mem_dim_model2, res_dim_model2)
model_2.set_in_state(np.array([[0., 0.], [0., 1.]]))

#Hamiltonian of model_2
h = aux_h_rand(sys_dim_model2*mem_dim_model2)
h = np.kron(h, np.eye(res_dim_model2))

#Hamiltonian of model_1
#TRUE HAMILTONINAN (model_1)
h_true = aux_h_rand(sys_dim_model1*mem_dim_model1)
h_true = np.kron(h_true, np.eye(res_dim_model1))
h_true = 0.8*h_true

dict_dimentions = {'s':sys_dim_model1,'m': mem_dim_model1,'r':res_dim_model1,'ism':sys_dim_model1*mem_dim_model1,\
'imr':mem_dim_model1*res_dim_model1,'isr': sys_dim_model1*res_dim_model1}
dict_h = {}
for j in dict_dimentions:
    dict_h[j] = aux_h_rand(dict_dimentions[j])

#Dimetion of h_isr = sys_dim_model1 * res_dim_model1#
#IT NEEDS TO PERFORM  OF "ALIGNMENT" or SORT OF INDECES IN MIXED HAMILTONIAN (h_isr) in the
#following order "SYSTEM"->"MEMORY"->"RESERVOIR"
# AT FIRST WE RESHAPE MULTIINDEX IN FORM OF SEPARATE INDECES, THEN WITH RESPECT TO PROPER
#ORDER WE EXPAND FULL DIMENTION OF h_isr
h_isr = np.kron(dict_h['isr'], np.eye(mem_dim_model1)).reshape(sys_dim_model1,res_dim_model1, \
mem_dim_model1,sys_dim_model1,res_dim_model1,mem_dim_model1) #SYSTEM Reservoir Memory#
h_isr = h_isr.transpose((0,2,1,3,5,4))

dict_h['isr'] = h_isr.reshape(sys_dim_model1 * res_dim_model1 * mem_dim_model1, \
sys_dim_model1 * res_dim_model1 * mem_dim_model1)

full_dimention = sys_dim_model1 * mem_dim_model1 * res_dim_model1


#RANDOMIZED model_1 Hamiltonian#
h_true = 0.6*np.kron(dict_h['s'], np.eye(mem_dim_model1 * res_dim_model1)) + 0.6*np.kron(np.eye(sys_dim_model1 * mem_dim_model1), dict_h['r'])\
 + 0.6*np.kron(np.kron(np.eye(sys_dim_model1), dict_h['m']), np.eye(res_dim_model1)) + 0.15*np.kron(np.eye(sys_dim_model1), dict_h['imr'])\
 + 0.3*np.kron(dict_h['ism'], np.eye(res_dim_model1)) + 0.1*dict_h['isr']
np.save('h_true.npy', h_true)
#Saving of model_1 Hamiltonian

#==================================PARAMETERS===================================
batch_size = 10**3
number_of_time_steps = 10**5
learning_rate = 0.001
number_of_epochs = 300
plotting_time_step = 0.1
max_total_plotting_time = 200
min_total_plotting_time = 20
#===============================================================================

max_time_steps = int(max_total_plotting_time/plotting_time_step)
min_time_steps = int(min_total_plotting_time/plotting_time_step)

#Hamiltonian setting
model_1.set_h(h_true)
model_2.set_h(h)

#Training set generation
tr_set = model_1.get_tr_set(number_of_time_steps)

tr_set_reshaped = tr_set.reshape((-1, batch_size, tr_set.shape[1], tr_set.shape[2])) #Splitting of training set
p1 = model_1.log_likelihood(tr_set) #True loglikelihood

#=========================LEARNING CICLE AND PLOTTING===========================
m = 0.
v = 0.
t = 1.
counter = 0
llh = np.array([])
true_llh = np.full(number_of_epochs, p1)

for epoch in range(number_of_epochs):
    llh_agr = 0.
    for batch in range(tr_set_reshaped.shape[0]):
        #raise Exception ('NICE')
        grad = model_2.grad(tr_set_reshaped[batch])
        m, v = model_2.adam_step(grad, learning_rate, m, v, t, b1=0.9, b2=0.95, epsilon=10.**(-4))
        t = t + 1.
        llh_agr = llh_agr + model_2.log_likelihood(tr_set_reshaped[batch])
    np.save('h.npy', model_2.h)
    llh = np.append(llh, llh_agr)
    fig1 = plt.figure()
    fig1.set_size_inches(10, 5)
    plt.plot(llh, 'b')
    plt.plot(true_llh, 'r')
    plt.legend(('Trained System','Real System'))
    plt.ylabel('Log Likelihood')
    plt.xlabel('Epochs')

    path = os.getcwd() + '/pic'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'pic/likelihood_fig{counter}.pdf')
    plt.pause(0.05)
    T = np.arange(0, max_total_plotting_time, plotting_time_step)
    rho1 = model_1.get_evolution_continuous(max_time_steps, plotting_time_step)
    rho2 = model_2.get_evolution_continuous(max_time_steps, plotting_time_step)
    model_1.set_in_state(np.array([[1., 0.], [0., 1.]])/2)
    model_2.set_in_state(np.array([[1., 0.], [0., 1.]])/2)
    rho_zero1 = model_1.get_evolution_continuous(max_time_steps, plotting_time_step)
    rho_zero2 = model_2.get_evolution_continuous(max_time_steps, plotting_time_step)
    model_1.set_in_state(np.array([[0., 0.], [0., 1.]]))
    model_2.set_in_state(np.array([[0., 0.], [0., 1.]]))
    sx1 = np.einsum('kij,ij->k', rho1, sigma[0])
    sx2 = np.einsum('kij,ij->k', rho2, sigma[0])
    sy1 = np.einsum('kij,ij->k', rho1, sigma[1])
    sy2 = np.einsum('kij,ij->k', rho2, sigma[1])
    sz1 = np.einsum('kij,ij->k', rho1, sigma[2])
    sz2 = np.einsum('kij,ij->k', rho2, sigma[2])
    fig2 = plt.figure()
    fig2.set_size_inches(10, 5)
    fig2.subplots_adjust(wspace=0.25, hspace=0.35)
    plt.subplot(2,2,1)
    plt.ylim(top = 1)
    plt.ylim(bottom = -1)
    plt.plot(T, sx1, 'r')
    plt.plot(T, sx2, 'b')
    plt.ylabel('$<\sigma_x>$')
    plt.subplot(2,2,2)
    plt.ylim(top = 1)
    plt.ylim(bottom = -1)
    plt.plot(T, sy1, 'r')
    plt.plot(T, sy2, 'b')
    plt.ylabel('$<\sigma_y>$')
    plt.subplot(2,2,3)
    plt.ylim(top = 1)
    plt.ylim(bottom = -1)
    plt.plot(T, sz1, 'r')
    plt.plot(T, sz2, 'b')
    plt.xlabel('Time')
    plt.ylabel('$<\sigma_z>$')
    plt.subplot(2,2,4)
    plt.ylim(top = 0.6)
    plt.ylim(bottom = 0)

    diff1 = rho1 - rho_zero1
    diff2 = rho2 - rho_zero2
    tr_dist1 = np.sqrt(np.einsum('kij,kij->k', diff1, diff1.conj()))/np.sqrt(2.)
    tr_dist2 = np.sqrt(np.einsum('kij,kij->k', diff2, diff2.conj()))/np.sqrt(2.)
    plt.plot(T, tr_dist1, 'r')
    plt.plot(T, tr_dist2, 'b')
    plt.ylabel('Trace distance')
    plt.xlabel('Time')
    plt.savefig(f'pic/long_dynamics_fig{counter}.pdf')
    fig3 = plt.figure()
    fig3.set_size_inches(10, 5)
    fig3.subplots_adjust(wspace=0.25, hspace=0.35)
    plt.subplot(2,2,1)
    plt.ylim(top = 1)
    plt.ylim(bottom = -1)
    plt.plot(T[:min_time_steps], sx1[:min_time_steps], 'r')
    plt.plot(T[:min_time_steps], sx2[:min_time_steps], 'b')
    plt.ylabel('$<\sigma_x>$')
    plt.subplot(2,2,2)
    plt.ylim(top = 1)
    plt.ylim(bottom = -1)
    plt.plot(T[:min_time_steps], sy1[:min_time_steps], 'r')
    plt.plot(T[:min_time_steps], sy2[:min_time_steps], 'b')
    plt.ylabel('$<\sigma_y>$')
    plt.subplot(2,2,3)
    plt.ylim(top = 1)
    plt.ylim(bottom = -1)
    plt.plot(T[:min_time_steps], sz1[:min_time_steps], 'r')
    plt.plot(T[:min_time_steps], sz2[:min_time_steps], 'b')
    plt.xlabel('Time')
    plt.ylabel('$<\sigma_z>$')
    plt.subplot(2,2,4)
    plt.ylim(top = 0.6)
    plt.ylim(bottom = 0)
    plt.plot(T[:min_time_steps], tr_dist1[:min_time_steps], 'r')
    plt.plot(T[:min_time_steps], tr_dist2[:min_time_steps], 'b')
    plt.ylabel('Trace distance')
    plt.xlabel('Time')
    plt.savefig(f'pic/dynamics_fig{counter}.pdf')
    counter = counter + 1
    plt.pause(0.05)
