from brian2 import *
import numpy as np
import time

# set up sine
arr = []
dt_test = .0001
period_in_dt = 40
times = (1 / (dt_test * period_in_dt) ) * 2 * np.pi * np.linspace(0, 1 + dt_test * period_in_dt, 1/dt_test)
arr = (10 * np.sin(times) + 1000) * Hz
ts = TimedArray(arr, dt=0.0001* second)

# constants, equations, detritus
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

# make the neurons
# ash = training neuron,
# input = input neurons looking at the past (see lags)
# neurons = output neuron
ash = PoissonGroup(1, rates='ts(t)', dt=0.0001 * second)
lags = [0, 4]
lags_ts = TimedArray(lags, dt = 1 * second)
input = PoissonGroup(len(lags),
        rates='ts(t - lags_ts(i * second) * 0.0001 * second)',
        dt=0.0001 * second)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='euler',dt=0.0001 * second)

# synapses
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
S.connect()
S.w = .01
S2 = Synapses(ash, neurons,
            '''w : 1''',
            on_pre='''ge += w ''',
            )
S2.connect()
S2.w = 0.01

# Monitors
sss = StateMonitor(S, variables=['w'], record=range(10000), dt=0.0001 * second)
mon = StateMonitor(neurons, variables = ['v'],record=range(10000), dt=0.0001 * second )

# Run and record
run(1*second, report='text')
list(map(print,sss.w[0]))
