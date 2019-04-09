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

lags = [0, 4]
# lags = [0, 2, 4, 8, 16]
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='euler',dt=0.0001 * second)
syn_statzi = []
neur_statzi = []
for prev in lags:
    input = PoissonGroup(1, rates='ts(t - {} * 0.0001 * second)'.format(prev),dt=0.0001 * second)
    if prev > 0:
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
    else:
        S = Synapses(input, neurons,
                    '''w : 1''',
                    on_pre='''ge += w ''',
                    )

    S.connect()
    S.w = .01
    # sss = StateMonitor(S, variables=['w', 'Apre', 'Apost'], record=range(10000))
    sss = StateMonitor(S, variables=['w'], record=range(10000), dt=0.0001 * second)
    mon = StateMonitor(neurons, variables = ['v'],record=range(10000), dt=0.0001 * second )
    syn_statzi.append(sss)
    neur_statzi.append(mon)

run(1*second, report='text')
print(', '.join('w: ' + str(i) for i in lags))
list(map(print, zip(*(s.w[0] for s in syn_statzi))))
print(syn_statzi)
for s in syn_statzi:
    plot(s.w[0])
    show()
