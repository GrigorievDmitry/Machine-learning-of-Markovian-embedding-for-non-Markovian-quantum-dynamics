import numpy as np
import numpy.random as rnd
import scipy.linalg as la
from scipy.stats import unitary_group as rnd_un

sigma = np.array([[[0., 1.],[1., 0.]], [[0., -1j],[1j, 0.]], [[1., 0.],[0., -1.]]])

def random_proj():
    
    x = rnd.normal(0, 1, 3)
    norm = la.norm(x)
    x = x/norm
    return 0.5*(np.einsum('i,ijk->jk',x,sigma) + np.eye(2))

def f(eig):
    eig = np.expand_dims(eig, axis = 0)
    minus = eig.T - eig
    plus = eig + eig.T
    return -1j*np.multiply(np.exp(-1j*plus/2), np.sinc(minus/(2*np.pi)))

def f_hc(eig):
    eig = np.expand_dims(eig, axis = 0)
    minus = eig.T - eig
    plus = eig + eig.T
    return 1j*np.multiply(np.exp(1j*plus/2), np.sinc(-minus/(2*np.pi)))
    
#def measurement(E, env):
#        return E.T.conj().dot(env.dot(E))

class dynamics_learning():
    
    def __init__(self, sys_dim=2, mem_dim=2, res_dim=2):
        self.sys_dim = sys_dim
        self.mem_dim = mem_dim
        self.res_dim = res_dim
        self.h = rnd.rand(res_dim*mem_dim*sys_dim, res_dim*mem_dim*sys_dim) + 1j*rnd.rand(res_dim*mem_dim*sys_dim, res_dim*mem_dim*sys_dim)
        self.h = 0.5*self.h + 0.5*np.conj(self.h.T)
        self.res_dens = np.eye(res_dim)
        self.res_dens[1:] = 0.
        self.mem_dens = np.eye(mem_dim)
        self.mem_dens[1:] = 0.
        self.sys_dens = np.eye(sys_dim)
        self.sys_dens[1:] = 0.
        self.u = la.expm(-1j * self.h)
        #resh_u = self.u.reshape(sys_dim, mem_dim, res_dim, sys_dim, mem_dim, res_dim)
        #self.phi = np.einsum('abcdef,fh,ijclmh -> abijdelm', resh_u, self.res_dens, resh_u.conj())
        #self.phi = self.phi.reshape(sys_dim * mem_dim, sys_dim * mem_dim, \
                                    #sys_dim * mem_dim, sys_dim * mem_dim)
        
    def set_h(self, h):
        self.h = h
        self.u = la.expm(-1j * h)
        #resh_u = self.u.reshape(self.sys_dim, self.mem_dim, self.res_dim, self.sys_dim, self.mem_dim, self.res_dim)
        #self.phi = np.einsum('abcdef,fh,ijclmh->abijdelm', resh_u, self.res_dens, resh_u.conj())
        #self.phi = self.phi.reshape(self.sys_dim * self.mem_dim, self.sys_dim * self.mem_dim, \
                                    #self.sys_dim * self.mem_dim, self.sys_dim * self.mem_dim)
                         
    def set_in_state(self, sys_dens):
        self.sys_dens = sys_dens
        
    def phi(self, state):
        whole_state = np.kron(state, self.res_dens)
        whole_state = self.u.dot(whole_state.dot(self.u.T.conj()))
        return np.einsum('ikjk->ij', whole_state.reshape(self.sys_dim*self.mem_dim, self.res_dim, self.sys_dim*self.mem_dim, self.res_dim))
    
    def rev_phi(self, env):
        whole_env = np.kron(env, np.eye(self.res_dim))
        whole_env = self.u.T.conj().dot(whole_env.dot(self.u))
        return np.einsum('ikjl,kl->ij', whole_env.reshape(self.sys_dim*self.mem_dim, self.res_dim, self.sys_dim*self.mem_dim, self.res_dim), self.res_dens)
    
    @staticmethod
    def measurement(E, env):
        return E.T.conj().dot(env.dot(E))
    
    def get_evolution_h_pert(self, total_time_steps, h_pert):
        state = np.kron(self.sys_dens, self.mem_dens) #KRON != TENSOR PRODUCT!!!! RESHAPE IN MATRIX!#
        set_of_states = np.expand_dims(self.sys_dens, axis=0)
        for i in range(1000):
            state = self.phi(state)
        resh_state = state.reshape((self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
        state = np.kron(self.sys_dens, np.einsum('ijkj->ik', resh_state))
        old_h = self.h
        self.set_h(old_h + h_pert)
        for i in range(total_time_steps - 1):
            state = self.phi(state)
            sys_state = np.einsum('ikjk->ij', state.reshape(self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
            set_of_states = np.append(set_of_states, np.expand_dims(sys_state, axis = 0),axis = 0)
        self.set_h(old_h)
            
        return set_of_states
    
    def get_evolution_h_pert_continuous(self, total_time_steps, time_step, h_pert):
        state = np.kron(self.sys_dens, self.mem_dens) #KRON != TENSOR PRODUCT!!!! RESHAPE IN MATRIX!#
        set_of_states = np.expand_dims(self.sys_dens, axis=0)
        for i in range(1000):
            state = self.phi(state)
        resh_state = state.reshape((self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
        state = np.kron(self.sys_dens, np.einsum('ijkj->ik', resh_state))
        old_h = self.h
        self.set_h(old_h + h_pert)
        u = self.u
        u = u.reshape((self.sys_dim, self.mem_dim, self.res_dim, self.sys_dim, self.mem_dim, self.res_dim))
        rho_res = self.res_dens
        ch = np.einsum('abcdef,fh,ijclmh->abijdelm', u, rho_res, u.conj())
        ch = ch.reshape((self.sys_dim*self.mem_dim*self.sys_dim*self.mem_dim, self.sys_dim*self.mem_dim*self.sys_dim*self.mem_dim))
        l = la.logm(ch)
        ch = la.expm(time_step*l)
        ch = ch.reshape((self.sys_dim*self.mem_dim, self.sys_dim*self.mem_dim, self.sys_dim*self.mem_dim, self.sys_dim*self.mem_dim))
        for i in range(total_time_steps - 1):
            state = np.einsum('ijkm,km->ij', ch, state)
            sys_state = np.einsum('ikjk->ij', state.reshape(self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
            set_of_states = np.append(set_of_states, np.expand_dims(sys_state, axis = 0),axis = 0)
        self.set_h(old_h)
            
        return set_of_states
    
    def get_evolution_continuous(self, total_time_steps, time_step):
        state = np.kron(self.sys_dens, self.mem_dens) #KRON != TENSOR PRODUCT!!!! RESHAPE IN MATRIX!#
        set_of_states = np.expand_dims(self.sys_dens, axis=0)
        for i in range(1000):
            state = self.phi(state)
        resh_state = state.reshape((self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
        state = np.kron(self.sys_dens, np.einsum('ijkj->ik', resh_state))
        u = self.u
        u = u.reshape((self.sys_dim, self.mem_dim, self.res_dim, self.sys_dim, self.mem_dim, self.res_dim))
        rho_res = self.res_dens
        ch = np.einsum('abcdef,fh,ijclmh->abijdelm', u, rho_res, u.conj())
        ch = ch.reshape((self.sys_dim*self.mem_dim*self.sys_dim*self.mem_dim, self.sys_dim*self.mem_dim*self.sys_dim*self.mem_dim))
        l = la.logm(ch)
        ch = la.expm(time_step*l)
        ch = ch.reshape((self.sys_dim*self.mem_dim, self.sys_dim*self.mem_dim, self.sys_dim*self.mem_dim, self.sys_dim*self.mem_dim))
        for i in range(total_time_steps - 1):
            state = np.einsum('ijkm,km->ij', ch, state)
            sys_state = np.einsum('ikjk->ij', state.reshape(self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
            set_of_states = np.append(set_of_states, np.expand_dims(sys_state, axis = 0),axis = 0)
            
        return set_of_states
    
    def get_evolution(self, total_time_steps):
        state = np.kron(self.sys_dens, self.mem_dens) #KRON != TENSOR PRODUCT!!!! RESHAPE IN MATRIX!#
        set_of_states = np.expand_dims(self.sys_dens, axis=0)
        for i in range(1000):
            state = self.phi(state)
        resh_state = state.reshape((self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
        state = np.kron(self.sys_dens, np.einsum('ijkj->ik', resh_state))
        for i in range(total_time_steps - 1):
            state = self.phi(state)
            sys_state = np.einsum('ikjk->ij', state.reshape(self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
            set_of_states = np.append(set_of_states, np.expand_dims(sys_state, axis = 0),axis = 0)
            
        return set_of_states
    
    def get_evolution_seq_h_pert(self, total_time_steps, h_pert_seq):
        state = np.kron(self.sys_dens, self.mem_dens) #KRON != TENSOR PRODUCT!!!! RESHAPE IN MATRIX!#
        set_of_states = np.expand_dims(self.sys_dens, axis=0)
        for i in range(1000):
            state = self.phi(state)
        resh_state = state.reshape((self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
        state = np.kron(self.sys_dens, np.einsum('ijkj->ik', resh_state))
        for i in range(total_time_steps - 1):
            old_h = self.h
            self.set_h(old_h + h_pert_seq[i])
            state = self.phi(state)
            sys_state = np.einsum('ikjk->ij', state.reshape(self.sys_dim, self.mem_dim, self.sys_dim, self.mem_dim))
            set_of_states = np.append(set_of_states, np.expand_dims(sys_state, axis = 0),axis = 0)
            self.set_h(old_h)
            
        return set_of_states
    
    def get_tr_set(self, total_time_steps):
        state = np.kron(self.sys_dens, self.mem_dens) #KRON != TENSOR PRODUCT!!!! RESHAPE IN MATRIX!#
        set_of_proj = np.empty((1, self.sys_dim * self.mem_dim, self.sys_dim * self.mem_dim), dtype = np.complex128)
        for i in range(total_time_steps):
            state = self.phi(state)
            proj = random_proj()
            proj1 = np.kron(proj, np.eye(self.mem_dim))
            test_state = dynamics_learning.measurement(proj1, state)
            p = np.trace(test_state)
            eta = rnd.rand()
            if p > eta:
                state = test_state/p
                set_of_proj = np.append(set_of_proj, np.expand_dims(proj1, axis = 0), axis = 0)
            else:
                proj2 = np.kron((np.eye(2) - proj), np.eye(self.mem_dim))
                state = dynamics_learning.measurement(proj2, state)/(1-p)
                set_of_proj = np.append(set_of_proj, np.expand_dims(proj2, axis = 0), axis = 0)
                
        return set_of_proj[1:]
    
    def log_likelihood(self, tr_set):
        llh = 0.
        state = np.kron(self.sys_dens, self.mem_dens)
        for e in tr_set:
            state = self.phi(state)
            state = dynamics_learning.measurement(e, state)
            p = np.trace(state)
            state = state/p
            llh += np.log(p)
        return llh
    
    def change_tr_set(self, tr_set):
        set_of_proj = np.empty((1, self.sys_dim * self.mem_dim, self.sys_dim * self.mem_dim), dtype = np.complex128)
        for e in tr_set:
            proj = np.einsum('ikjk->ij', e.reshape(2, self.mem_dim, 2, self.mem_dim))/self.mem_dim
            eta = rnd.choice(np.array([0, 1]))
            if eta == 1:
                set_of_proj = np.append(set_of_proj, np.expand_dims(np.kron(proj, np.eye(self.mem_dim)), axis = 0), axis = 0)
            else:
                set_of_proj = np.append(set_of_proj, np.expand_dims(np.kron(np.eye(2) - proj, np.eye(self.mem_dim)), axis = 0), axis = 0)
        return set_of_proj[1:]
    
    def grad(self, tr_set):
        val, vec = la.eig(self.h)
        num = tr_set.shape[0]
        tr_set_flip = np.flip(tr_set, 0)
        env = dynamics_learning.measurement(tr_set_flip[0], np.eye(self.sys_dim*self.mem_dim))
        rho = np.kron(self.sys_dens, self.mem_dens)
        set_of_rho = np.expand_dims(rho, axis = 0)
        set_of_env = np.expand_dims(env, axis = 0)
        u = self.u
        u_plus = u.T.conj()
        grd = np.zeros((self.sys_dim*self.mem_dim*self.res_dim, self.sys_dim*self.mem_dim*self.res_dim))
        for i in range(num - 1):
            rho = self.phi(rho)
            rho = dynamics_learning.measurement(tr_set[i], rho)
            rho = rho/np.trace(rho)
            set_of_rho = np.append(set_of_rho, np.expand_dims(rho, axis = 0), axis = 0)
            
            env = self.rev_phi(env)
            env = dynamics_learning.measurement(tr_set_flip[i+1], env)
            env = env/np.trace(env)
            set_of_env = np.append(set_of_env, np.expand_dims(env, axis = 0), axis = 0)
        set_of_env = np.flip(set_of_env, 0)
        for i in range(num):
            norm = np.trace(set_of_env[i].dot(self.phi(set_of_rho[i])))
            env1 = np.kron(set_of_rho[i], self.res_dens).dot(u_plus.dot(np.kron(set_of_env[i], np.eye(self.res_dim))))
            env2 = np.kron(set_of_env[i], np.eye(self.res_dim)).dot(u.dot(np.kron(set_of_rho[i], self.res_dens)))
            env1 = np.multiply(vec.T.conj().dot(env1.dot(vec)), f(val).T)
            env2 = np.multiply(vec.T.conj().dot(env2.dot(vec)), f_hc(val).T)
            env1 = vec.dot(env1.dot(vec.T.conj()))
            env2 = vec.dot(env2.dot(vec.T.conj()))
            env1 = env1 + env1.conj().T
            env2 = env2 + env2.conj().T
            update = (env1 + env2)/norm
            grd = grd + update
            
        return grd/2.
    
    def adam_step(self, grad, learning_rate, m, v, t, b1=0.9, b2=0.999, epsilon=10.**(-8)):
        m_new = b1*m + (1 - b1)*grad
        v_new = b2*v + (1 - b2)*np.square(grad)
        m_cor = m_new/(1 - b1**t)
        v_cor = v_new/(1 - b2**t)
        self.set_h(self.h + learning_rate*(m_cor/(np.sqrt(v_cor + epsilon))))
        return m_new, v_new
    
    def momentum_step(self, grad, learning_rate, m, gamma=0.9):
        m_new = gamma*m + (1 - gamma)*grad
        self.set_h(self.h + learning_rate*m_new)
        return m_new
        