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
tr_set = model_1.get_tr_set(1000)
p1 = model_1.log_likelihood(tr_set)

true_llh = np.full(30, p1)


llh_set = np.array([])
counter = 0

def cbk(params):
    llh = model_2.log_likelihood(tr_set)
    global llh_set
    global counter
    llh_set = np.append(llh_set, llh)
    fig1 = plt.figure()
    fig1.set_size_inches(10, 5)
    plt.plot(llh_set, 'b')
    plt.plot(true_llh, 'r')
    plt.legend(('Trained System','Real System'))
    plt.ylabel('Log Likelihood')
    plt.xlabel('Iterations')
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
    
h_re = np.real(h)
h_im = np.imag(h)
hh = np.append(np.expand_dims(h_re, axis=0), np.expand_dims(h_im, axis=0), axis=0)
shp = hh.shape
hh = hh.reshape(128)
def f(hh):
    hh = hh.reshape(shp)
    hre = hh[0]
    him = hh[1]
    model_2.set_h(hre + 1j*him)
    p = -model_2.log_likelihood(tr_set)
    return p

def grd(hh):
    hh = hh.reshape(shp)
    hre = hh[0]
    him = hh[1]
    model_2.set_h(hre + 1j*him)
    g = -model_2.grad(tr_set)
    gre = np.real(g)
    gim = np.imag(g)
    g = np.append(np.expand_dims(gre, axis=0), np.expand_dims(gim, axis=0), axis=0).reshape(128)
    return g

x, f, inform = bfgs(f, hh, grd, maxiter=30, callback=cbk)
print(inform['nit'])

h_true = model_1.h
h = model_2.h
h_pert = np.kron(rnd.rand(2,2) + 1j*rnd.rand(2,2), np.eye(4,4))
h_pert = 0.5*(h_pert + h_pert.conj().T)
h_true = h_true + h_pert
h = h + h_pert
model_1.set_h(h_true)
model_2.set_h(h)

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
fig3 = plt.figure()
fig3.set_size_inches(10, 5)
fig3.subplots_adjust(wspace=0.25, hspace=0.35)
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
plt.savefig(f'pic/perturbation{counter}.pdf')