import cmath #To help us out with the complex square root
import numpy as np #For the arrays
import matplotlib.pyplot as plt #Visualization
mass = 1.0 #Mass, one for simplicity
hbar = 1.0 #HBar, one for simplicity
v0 = 2.0 #Initial potential value
alpha = 0.5 #Value for our potential equation
E = 3.0 #Energy
i = 1.0j #Defining imaginary number
x = 10.0 #Initial x-value
xf = -10.0 #Final x-value
h = -.001 #Step value
xaxis = np.array([], float) #Empty array to fill with out x-values
psi = np.array([], complex) #Empty array to fill with the values for the initial equation we are trying to solve, defined array as complex to fill with complex numbers
psiprime = np.array([], complex) #Empty array to fill with the values for the first derivative equation we are trying to solve, defined array as complex to fill with complex numbers
def v(x): #Potential equation we will be using for this example
    return v0/2.0 * (1.0 + np.tanh(x/alpha))
def k(x): #Reworked Schrödinger's equation to solve for k
    return cmath.sqrt((2*mass/(hbar**2))*(E - v(x)))
def psione(x): #PSI, wavefunction equation
    return np.exp(i*k(x)*x)

def psitwo(x): #Derivative of the psione equation
    return i*k(x)*np.exp(i*k(x)*x)
r = np.array([psione(x), psitwo(x)]) #Array with wavefunctions, usually this is where our initial condition equations go.

def deriv(r,x):
    return np.array([r[1],-(2.0*mass/(hbar**2) * (E - v(x))*r[0])], complex)

#The double star, **, is for exponents
#While loop to iterate through the Runge-Kutta. This particular version, the Fourth Order, will have four slope values that help approximate then next slope value, from k1 to k2, k2 to k3, and k3 to k4.
#This loop also appends that values, starting with the initial values, to the empty arrays that we've initialized earlier.
while (x >= xf ):
    xaxis = np.append(xaxis, x)
    psi = np.append(psi, r[0])
    psiprime = np.append(psiprime, r[1])
    k1 = h*deriv(r,x)
    k2 = h*deriv(r+k1/2,x+h/2)
    k3 = h*deriv(r+k2/2,x+h/2)
    k4 = h*deriv(r+k3,x+h)
    r += (k1+2*k2+2*k3+k4)/6
    x += h #The += in this line, and the line above, is the same thing as telling the code to x = x + h, which updates x, using the previous x with the addition of the step value.

#Grabbing the last values of the arrays and redefining our x-axis
psi1 = psi[20000]; psi2 = psiprime[20000]; x = 10; xf = -10
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
#Outputs for the print command
#reflection = 0.007625630800891285
#transmission = (0.9923743691991354+0j)
#r + t = (1.0000000000000266+0j)
#Ideally, r + t should give us one, a bit stumped if the precision that's present in Python can lead to the small discrepancy, without considering formatting the answer to a set amount of decimal values.
#Plotting the graphs side by side, including the imaginary values.
fig, ax = plt.subplots(1,2, figsize = (15,5))
ax[0].plot(xaxis, psi.real, xaxis, psi.imag, xaxis, v(xaxis))
ax[1].plot(xaxis, psiprime.real, xaxis, psiprime.imag, xaxis, v(xaxis))
plt.show()