import numpy as np
import matplotlib.pyplot as plt

N = 500
dims = 3
input_variance = np.sqrt(0.002)
reservoir_variance = np.sqrt(2/N)
k = 0.01
time_steps = 500

w_input = np.random.normal(loc=0, scale=input_variance, size=(N, dims))
w_reservoir = np.random.normal(loc=0, scale=reservoir_variance, size=(N, N))
r = np.zeros(N)
x_train = np.genfromtxt('training-set.csv', delimiter=',')
x_test = np.genfromtxt('test-set-5.csv', delimiter=',')
size_train = x_train.shape[1]
size_test = x_test.shape[1]
R = np.zeros((N, size_train))
o = np.zeros((dims, time_steps))


def train_reservoir(_w_reservoir, _w_input, _r, _R, _x_train, _size_train):
    for t in range(size_train):
        _R[:, t] = _r
        _r = np.tanh(np.dot(_w_reservoir, _r) + np.dot(_w_input, _x_train[:, t]))
    _R[:, -1] = _r
    return _R


def optimize(_R, _x_train, _k, _N):
    _w_out = np.dot(np.dot(_x_train, _R.transpose()), np.linalg.inv(np.dot(_R, _R.transpose()) + k * np.identity(_N)))
    return _w_out


def output(_w_reservoir, _w_input, _w_out, _x_test, _o, _r, _size_test, _time_steps):
    for t in range((_size_test - 1)):
        _r = np.tanh(np.dot(_w_reservoir, _r) + np.dot(_w_input, _x_test[:, t]))
    for t in range(_time_steps):
        _o[:, t] = np.dot(_w_out, _r)
        _r = np.tanh(np.dot(_w_reservoir, _r) + np.dot(_w_input, _o[:, t]))
    return _o


R_out = train_reservoir(w_reservoir, w_input, r, R, x_train, size_train)
w_out = optimize(R_out, x_train, k, N)
o = output(w_reservoir, w_input, w_out, x_test, o, r, size_test, time_steps)
# Save y-component
np.savetxt('prediction.csv', o[1, :], delimiter=',')

# Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.plot(o[0, :], o[1, :], o[2, :])

plt.show()
