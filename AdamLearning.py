import numpy as np
import numpy.random as rnd
import scipy.linalg as lg
import matplotlib.pyplot as plt
import dynamics_learning as dl
from dynamics_learning import dynamics_learning as dl1
from scipy.optimize import fmin_l_bfgs_b as bfgs

sigma = np.array([[[0., 1.],[1., 0.]], [[0., -1j],[1j, 0.]], [[1., 0.],[0., -1.]]])



model_1 = dl.dynamics_learning(2, 2, 2)
model_2 = dl.dynamics_learning(2, 2, 2)
model_1.set_in_state(np.array([[0., 0.], [0., 1.]]))
model_2.set_in_state(np.array([[0., 0.], [0., 1.]]))
h = rnd.rand(4,4) + 1j*rnd.rand(4,4)
h = np.kron(h + h.conj().T, np.eye(2))
#h_true = rnd.rand(4,4) + 1j*rnd.rand(4,4)
#h_true = h_true + h_true.conj().T
#h_true = np.kron(h_true + h_true.conj().T, np.eye(2))
#h_true = 0.8*h_true
h_s = rnd.rand(2,2) + 1j*rnd.rand(2,2)
h_s = h_s + h_s.T.conj()
h_m = rnd.rand(2,2) + 1j*rnd.rand(2,2)
h_m = h_s + h_s.T.conj()
h_r = rnd.rand(2,2) + 1j*rnd.rand(2,2)
h_r = h_s + h_s.T.conj()
h_ism = rnd.rand(4,4) + 1j*rnd.rand(4,4)
h_ism = h_ism + h_ism.T.conj()
h_imr = rnd.rand(4,4) + 1j*rnd.rand(4,4)
h_imr = h_imr + h_imr.T.conj()
h_isr = rnd.rand(4,4) + 1j*rnd.rand(4,4)
h_isr = h_isr + h_isr.T.conj()
h_isr = np.kron(h_isr, np.eye(2)).reshape((2,2,2,2,2,2))
h_isr = h_isr.transpose((0,2,1,3,5,4))
h_isr = h_isr.reshape((8,8))
h_true = 0.6*np.kron(h_s, np.eye(4)) + 0.6*np.kron(np.eye(4), h_r)\
 + 0.6*np.kron(np.kron(np.eye(2), h_m), np.eye(2)) + 0.15*np.kron(np.eye(2), h_imr)\
 + 0.3*np.kron(h_ism, np.eye(2)) + 0.1*h_isr
model_1.set_h(h_true)
model_2.set_h(h)
tr_set = model_1.get_tr_set(100000)
batch_size = 1000
tr_set_reshaped = tr_set.reshape((-1, batch_size, tr_set.shape[1], tr_set.shape[2]))
p1 = model_1.log_likelihood(tr_set)
m = 0.
v = 0.
t = 1.
learning_rate = 0.001
counter = 0
llh = np.array([])
true_llh = np.full(300, p1)

for epoch in range(300):
    llh_agr = 0.
    for batch in range(tr_set_reshaped.shape[0]):
        grad = model_2.grad(tr_set_reshaped[batch])
        m, v = model_2.adam_step(grad, learning_rate, m, v, t, b1=0.9, b2=0.95, epsilon=10.**(-4))
        t = t + 1.
        llh_agr = llh_agr + model_2.log_likelihood(tr_set_reshaped[batch])
    llh = np.append(llh, llh_agr)
    fig1 = plt.figure()
    fig1.set_size_inches(10, 5)
    plt.plot(llh, 'b')
    plt.plot(true_llh, 'r')
    plt.legend(('Trained System','Real System'))
    plt.ylabel('Log Likelihood')
    plt.xlabel('Epochs')
    plt.savefig(f'pic/likelihood_fig{counter}.png')
    plt.pause(0.05)
    rho1 = model_1.get_evolution(20)
    rho2 = model_2.get_evolution(20)
    model_1.set_in_state(np.array([[1., 0.], [0., 1.]])/2)
    model_2.set_in_state(np.array([[1., 0.], [0., 1.]])/2)
    rho_zero1 = model_1.get_evolution(20)
    rho_zero2 = model_2.get_evolution(20)
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
    plt.plot(sx1, 'r')
    plt.plot(sx2, 'b')
    plt.ylabel('$<\sigma_x>$')
    plt.subplot(2,2,2)
    plt.plot(sy1, 'r')
    plt.plot(sy2, 'b')
    plt.ylabel('$<\sigma_y>$')
    plt.subplot(2,2,3)
    plt.plot(sz1, 'r')
    plt.plot(sz2, 'b')
    plt.xlabel('Time')
    plt.ylabel('$<\sigma_z>$')
    plt.subplot(2,2,4)
    diff1 = rho1 - rho_zero1
    diff2 = rho2 - rho_zero2
    tr_dist1 = np.sqrt(np.einsum('kij,kij->k', diff1, diff1.conj()))/np.sqrt(2.)
    tr_dist2 = np.sqrt(np.einsum('kij,kij->k', diff2, diff2.conj()))/np.sqrt(2.)
    plt.plot(tr_dist1, 'r')
    plt.plot(tr_dist2, 'b')
    plt.ylabel('Trace distance')
    plt.xlabel('Time')
    plt.savefig(f'pic/dynamics_fig{counter}.png')
    counter = counter + 1
    plt.pause(0.05)