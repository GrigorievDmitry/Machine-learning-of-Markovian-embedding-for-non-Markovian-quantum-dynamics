# Quantum_dynamics
A code for extracting hidden features of the quantum reservoir via projective measurements on the controlled system. AdamLearning.py provides verification of the method descibed in Ref.[arXiv]. Dynamics_learning.py provides following list of functions which can be used on practice (!!!current version of the code works well only with sys_dim=2!!!):
1) Object which contains parameters of system (Hamiltonian, initial state, dimension of subspaces) can be created by following line of code:
```python
dynamics_learning(sys_dim=2, mem_dim=2, res_dim=2)
```
2) This object has list of methods:
* Set Hamiltonian h:
```python
set_h(h)
```
* Set initial system state:
```python
set_in_state(sys_dens)
```
* Get evolution of system with fixed time step(dt=1) and given number of time steps:
```python
get_evolution(total_time_steps)
```
* Get evolution of system with perturbated hamiltonian (h + h_pert) fixed time step(dt=1) and given number of time steps:
```python
get_evolution_h_pert(total_time_steps, h_pert)
```
* Get evolution of system with perturbated hamiltonian (h+h_pert) given time step (dt=time_step) and given number of time: steps
```python
get_evolution_h_pert_continuous(total_time_steps, time_step, h_pert)
```
* Get evolution of system with given time step (dt=time_step) and given number of time steps:
```python
get_evolution_continuous(total_time_steps, time_step)
```
* Get traning set from the model:
```python
get_tr_set(total_time_steps)
```
* Get logarithmic likelihood with given training set:
```python
log_likelihood(tr_set)
```
* Get gradient w.r.t Hamiltonian with given training set:
```python
grad(tr_set)
```
* One step of optimization with Adam algorithm:
```python
adam_step(grad, learning_rate, m, v, t, b1=0.9, b2=0.999, epsilon=10.**(-8))
```
* One step of optimization with Momentum SGD algorithm:
```python
momentum_step(grad, learning_rate, m, gamma=0.9)
```
