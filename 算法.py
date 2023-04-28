import numpy as np
from scipy.linalg import solve_banded
from scipy.special import gamma
import matplotlib.pyplot as plt
import time

X = 1.0;
T = 1.0;
Nx = 50;
Nt = 50;
dx = X / Nx;
dt = T / Nt
tArray = np.linspace(0, T, Nt + 1)
xArray = np.linspace(0, X, Nx + 1)


def f(x):  # spatial varying source
    #return np.sin(np.pi*x)
    return x - x ** 2
    #return 1 - abs(2*x-1)


def g(t):  # temporally varying source
    return np.sin(np.pi * t)
    # return 10*(t-t**2)
    # return 1 - abs(2*t-1)


f_true = f(xArray)  # change the function f(x) to np.array
g_true = g(tArray)  # change the function g(t) to np.array
def IBVP_frac(alpha, initial, source):
    # "alpha": fractional order; "initial" and "source" are ndarrays
    # priori setting for solution u
    U = np.zeros((Nx + 1, Nt + 1))
    b = np.zeros(Nt)
    # governing matrix
    p = gamma(2 - alpha) * dt ** alpha
    rho = p / dx ** 2
    for k in range(Nt):
        b[k] = (k + 1) ** (1 - alpha) - k ** (1 - alpha)
    mat_diag = np.array([-np.ones(Nt - 1) * rho, np.ones(Nt - 1) * (1.0 + 2. * rho), -np.ones(Nt - 1) * rho])

    # setting the initial value
    U[:, 0] = initial

    # working out the first level
    rhs = initial[1:Nx] + p * source[1:Nx, 1]
    x = solve_banded((1, 1), mat_diag, rhs)
    U[1:Nx, 1] = x

    # for the case of k>0 in the discrete form Ax=b
    for k in range(1, Nt):
        s = 0
        for i in range(k):
            s = s + (b[i] - b[i + 1]) * U[1:Nx, k - i]
        rhs = b[k] * initial[1:Nx] + p * source[1:Nx, k + 1] + s
        x = solve_banded((1, 1), mat_diag, rhs)
        U[1:Nx, k + 1] = x
    return U


def Inte_time(U):
    return np.sum(U, axis=1) * dt


def iter_isp(alpha, noise):
    stop_erro = noise
    iter_step = 0
    f_iter = np.zeros(Nx + 1)
    c = np.zeros(Nt + 1)
    for i in range(Nt+1):
        c[i] = dt ** (1 - alpha) * ((Nt + 1 - i) ** (1 - alpha) - (Nt + 1 - i - 1) ** (1 - alpha))
    source_true = np.array([f(x) * g(t) for x in xArray for t in tArray]).reshape(-1, Nt + 1)
    sol_true = IBVP_frac(alpha, np.zeros(Nx + 1), source_true)
    diff1=np.diff(sol_true)*dx**(-1)
    diff2=np.diff(diff1)*dx**(-1)
    rand_noise = noise * 2 * (np.random.rand(Nx + 1) - 0.5)
    psi = -Inte_time(diff2)
    psi =psi + np.multiply(psi, rand_noise)
    while iter_step <= 100:
        ff_iter = f_iter
        source_ff_iter = np.array([x * g(t) for x in ff_iter for t in tArray]).reshape(-1, Nt + 1)
        u_ff = IBVP_frac(alpha, np.zeros(Nx + 1), source_ff_iter)
        f_iter = (gamma(2 - alpha) ** (-1)) * np.dot(u_ff, c) + psi
        abs_erro = np.linalg.norm(ff_iter - f_iter) / np.linalg.norm(ff_iter)
        real_erro = np.linalg.norm(f_iter - f_true) / np.linalg.norm(f_true)
        iter_step += 1
        if abs_erro < stop_erro:
            break
    print('The steps of iteration is', iter_step)
    print('The real error is', real_erro)
    return f_iter
alpha = 0.1;
noise = 0.001
plt.figure()
plt.plot(xArray, f_true, 'r-', label='exact solution')
plt.plot(xArray, iter_isp(alpha, noise), 'g*', label='noise=%.3f' % noise)
# plt.plot(xArray, iter_isp(0.05), 'b--', label = 'noise=0.05')
plt.xlabel(u'$x$', fontproperties='SimHei', fontsize=14)
plt.ylabel(u'$f(x)$', fontproperties='SimHei', fontsize=14)
plt.legend()
plt.show()