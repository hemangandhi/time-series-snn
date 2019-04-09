from brian2 import *
import numpy as np
import time

arr = []
dt_test = .0001
period_in_dt = 40
times = (1 / (dt_test * period_in_dt) ) * 2 * np.pi * np.linspace(0, 1 + dt_test * period_in_dt, 1/dt_test)
arr = (10 * np.sin(times) + 1000) * Hz
ts = TimedArray(arr, dt=0.0001* second)

N = 1000
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
F = 15*Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax
ged = 1

eqs_neurons = '''
dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

input = PoissonGroup(1, rates='ts(t - 0.0004 * second)',dt=0.0001 * second)
ash = PoissonGroup(1, rates='ts(t)',dt=0.0001 * second)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='euler',dt=0.0001 * second)
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )
S2  = Synapses(ash, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w ''')
S2.connect()
S.connect()
S.w = .01
S2.w = .01
sss = StateMonitor(S, variables=['w', 'Apre', 'Apost'], record=range(10000))
mon = StateMonitor(neurons, variables = ['v', 'ge'],record=range(10000),dt=0.0001 * second )
s_mon = SpikeMonitor(neurons,variables = ['v'])

run(1*second, report='text')
print(mon.t / second, mon.v[0] / volt)
print('w, pre, post', 'sin')
list(map(print, zip(sss[0].w, sss[0].Apre, sss[0].Apost, arr)))
# plot(mon.t / second, mon.v[0] / mV)
plot(mon.t / second, arr - 1000 * Hz)
plot(sss.t / second, 100 * sss.w[0], color='red')
show()
