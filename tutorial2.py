import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
E = 3; m = 1; h = 1; alpha = .5; v0=2; i = 1.0j; xi = 10; xf = -10
def v(x): return v0/2.0 * (1.0 + np.tanh(x/alpha))
def k(x): return cmath.sqrt((2*m/(h**2))*(E - v(x)))
def psione(x): return np.exp(i*k(x)*x)
def psitwo(x): return i*k(x)*np.exp(i*k(x)*x)
def deriv(x, y): return [y[1], -(2.0*m/(h**2.0) * (E - v(x))*y[0])]
# solve_ivp is a built in rk45step solver
values = solve_ivp(deriv, [10, -10], [psione(xi), psitwo(xi)],
first_step = .001, max_step = .001)
psi1 = values.y[0,20000]; psi2 = values.y[1,20000]; x = 10; xf = -10
def reflection(x, y):
    aa = (psi1 + psi2/(i*k(y)))/(2*np.exp(i*k(y)*y))
    bb = (psi1 - psi2/(i*k(y)))/(2*np.exp(-i*k(y)*y))
    return (np.abs(bb)/np.abs(aa))**2
def transmission(x,y):
    aa = (psi1 + psi2/(i*k(y)))/(2.0*np.exp(i*k(y)*y))
    return k(x)/k(y) * 1.0/(np.abs(aa))**2
print('reflection = ',reflection(x,xf))
print('transmission = ', transmission(x,xf))
print('r + t = ', reflection(x,xf) + transmission(x,xf))
fig, ax = plt.subplots(1,2, figsize = (15,5))
ax[0].plot(values.t, values.y[0].real, values.t, values.y[0].imag, values.t, v(values.t))
ax[1].plot(values.t, values.y[1].real, values.t, values.y[1].imag, values.t, v(values.t))
plt.show()