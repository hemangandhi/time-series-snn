from brian2 import *
import numpy as np
import time

def make_x(period_in_dt=21, x_scale=1, dt_test=.0001):
    times = x_scale * np.linspace(0, 1 + dt_test * period_in_dt, 1/dt_test) * Hz
    return times

def make_sine(period_in_dt=21, dt_test=.0001):
    # set up sine
    times = (1 / (dt_test * period_in_dt) ) * 2 * np.pi * np.linspace(0, 1 + dt_test * period_in_dt, 1/dt_test)
    arr = (10 * np.sin(times) + 1000) * Hz
    return arr

def make_snn_and_run(ts, aggregator, lags=[2, 3, 5], duration=1*second, dt_ts=0.0001 * second, runs=1):
    # constants, equations, detritus
    ts = TimedArray(ts, dt=dt_ts)
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
    # mon = StateMonitor(neurons, variables = ['v'],record=range(10000), dt=0.0001 * second )

    # Run and record
    for r in range(runs):
        run(duration, report='text')
        aggregator(r, sss)
        #TODO: this is a resource issue otherwise?
        #also, what do >1 second runs mean?
        #and how reset SNN?
        sss.resize(0)
        print("run {} done".format(r))

def print_statzi(nv):
    def aggie(i, sss):
        list(map(print,zip(*sss.w[0:nv])))
    return aggie

def plot_statzi(lags):
    def aggie(i, sss):
        list(map(lambda x: plt.plot(x[0], label=x[1]),zip(sss.w[0:len(lags)], ("t - {}".format(l) for l in lags))))
        plt.legend()
        plt.show()
    return aggie

if __name__ == "__main__":
    # plot_statzi(make_snn_and_run(make_x()), 3)
    # plot_statzi(make_snn_and_run(make_sine()), 3)
    x_plus_sin = make_x(x_scale=5) + make_sine()
    guy_fawkes = plot_statzi([2, 3, 5])
    make_snn_and_run(x_plus_sin, guy_fawkes, runs=3)
