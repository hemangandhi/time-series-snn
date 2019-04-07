import numpy as np
import matplotlib.pyplot as plt

from brian2 import *

class Sine:
    def __init__(self, amplitude, freq, phase_shift, y_shift):
        self.a = amplitude
        self.p = freq
        self.s = phase_shift
        self.y = y_shift
    def __call__(self, arg):
        return np.sin(self.p * arg + self.s) * self.a + self.y
    def time_series(self, start, stop, num):
        return self(np.linspace(start, stop, num=num))
    def period(self):
        return 2 * np.pi / self.p
    def to_brian_fn(self, free_var):
        return "({} * cos(int({} * {} / second) + {})) * mV".format(self.a * self.p, self.p, free_var, self.s)

def stdp_experiment(timeseries):
    N = 1000
    taum = 10*ms
    taupost = taupre = 20*ms
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

    #lif, ge is apparently the internal charge
    eqs_neurons = '''
    dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    '''

    #TODO: make sure timeseries actually is input current?
    # Also, multiple inputs with each off by an amount of time.
    input = NeuronGroup(1, 'v = timeseries(t): volt', threshold='v>vt', method='exact')
    #Hidden LIF bois
    neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                        method='exact')
    S = Synapses(input, neurons, #question: what are Apre and Apost initialized to?
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
    S.w = 'rand() * gmax'
    mon = StateMonitor(S, 'w', record=[0, 1])
    s_mon = SpikeMonitor(input)

    run(100*second, report='text')

if __name__ == "__main__":
    one_over_dt = 100
    stdp_experiment(TimedArray(Sine(1, 1, 0, 0).time_series(0, 1, one_over_dt) * mV, 1/one_over_dt * second)) #the virgin sine
