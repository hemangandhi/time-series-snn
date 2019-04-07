from brian2 import *
import numpy as np
import time

arr = []
dt_test = .0001
arr = (np.sin(np.linspace(0, 1, 10000)) + 1000) * Hz
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


input = PoissonGroup(1, rates='ts(t)',dt=0.0001 * second)
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
S2  = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w ''')
S2.connect()
S.connect()
S.w = 1
S2.w = 1
mon = StateMonitor(neurons, variables = ['v', 'ge'],record=range(10000),dt=0.0001 * second )

s_mon = SpikeMonitor(neurons,variables = ['v'])

run(1*second, report='text')
print(mon.t, mon.v)
plot(mon.t/ second, mon.v[0]/ volt )
show()



