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
p1 = model_1.log_likelihood(tr_set)
m = 0.
v = 0.
llh = np.array([])
true_llh = np.full(100, p1)
"""
for k in range(10000):
    ind = rnd.randint(9899)
    batch = tr_set[ind:ind+100]
    grad = model_2.grad(batch)
    m = model_2.momentum_step(grad, 0.02, m)
    if k%100 == 0:
        llh = np.append(llh, model_2.log_likelihood(tr_set))
        fig1 = plt.figure()
        plt.plot(llh)
        plt.plot(true_llh)
        fig2 = plt.figure()
        rho1 = model_1.get_evolution(20)
        rho2 = model_2.get_evolution(20)
        sz1 = np.einsum('kij,ij->k', rho1, sigma[2])
        sz2 = np.einsum('kij,ij->k', rho2, sigma[2])
        plt.plot(sz1)
        plt.plot(sz2)
        plt.pause(0.05)
"""
"""
for k in range(100):
    grad = model_2.grad(tr_set)
    m, v = model_2.adam_step(grad, 0.05, m, v)
    llh = np.append(llh, model_2.log_likelihood(tr_set))
    fig1 = plt.figure()
    plt.plot(llh)
    plt.plot(true_llh)
    fig2 = plt.figure()
    rho1 = model_1.get_evolution(20)
    rho2 = model_2.get_evolution(20)
    sz1 = np.einsum('kij,ij->k', rho1, sigma[2])
    sz2 = np.einsum('kij,ij->k', rho2, sigma[2])
    plt.plot(sz1)
    plt.plot(sz2)
    plt.pause(0.05)
"""
"""
for k in range(1000):
    grad = model_2.grad(tr_set)
    m = model_2.momentum_step(grad, 0.003, m)
    llh = np.append(llh, model_2.log_likelihood(tr_set))
    fig1 = plt.figure()
    plt.plot(llh)
    plt.plot(true_llh)
    fig2 = plt.figure()
    rho1 = model_1.get_evolution(20)
    rho2 = model_2.get_evolution(20)
    sx1 = np.einsum('kij,ij->k', rho1, sigma[0])
    sx2 = np.einsum('kij,ij->k', rho2, sigma[0])
    sy1 = np.einsum('kij,ij->k', rho1, sigma[1])
    sy2 = np.einsum('kij,ij->k', rho2, sigma[1])
    sz1 = np.einsum('kij,ij->k', rho1, sigma[2])
    sz2 = np.einsum('kij,ij->k', rho2, sigma[2])
    plt.plot(sz1)
    plt.plot(sz2)
    fig3 = plt.figure()
    plt.plot(np.sqrt(sx1**2 + sy1**2 + sz1**2))
    plt.plot(np.sqrt(sx2**2 + sy2**2 + sz2**2))
    plt.pause(0.05)
"""

llh_set = np.array([])
counter = 0

def cbk(params):
    llh = model_2.log_likelihood(tr_set)
    global llh_set
    global counter
    llh_set = np.append(llh_set, llh)
    fig = plt.figure()
    fig.set_size_inches(6, 10)
    plt.subplot(3,1,1)
    plt.plot(llh_set, 'b')
    plt.plot(true_llh, 'r')
    plt.legend(('Trained System','Real System'))
    plt.ylabel('Log Likelihood')
    plt.subplot(3,1,2)
    rho1 = model_1.get_evolution(20)
    rho2 = model_2.get_evolution(20)
    sx1 = np.einsum('kij,ij->k', rho1, sigma[0])
    sx2 = np.einsum('kij,ij->k', rho2, sigma[0])
    sy1 = np.einsum('kij,ij->k', rho1, sigma[1])
    sy2 = np.einsum('kij,ij->k', rho2, sigma[1])
    sz1 = np.einsum('kij,ij->k', rho1, sigma[2])
    sz2 = np.einsum('kij,ij->k', rho2, sigma[2])
    plt.plot(sz1, 'r')
    plt.plot(sz2, 'b')
    plt.ylabel('$<\sigma_z>$')
    plt.subplot(3,1,3)
    plt.plot(np.sqrt(sx1**2 + sy1**2 + sz1**2), 'r')
    plt.plot(np.sqrt(sx2**2 + sy2**2 + sz2**2), 'b')
    plt.ylabel('$r$')
    plt.xlabel('Iteration')
    plt.savefig(f'pic/fig{counter}.png')
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

x, f, inform = bfgs(f, hh, grd, maxiter=100, callback=cbk)
print(inform['nit'])