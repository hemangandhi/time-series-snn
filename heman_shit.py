from brian2 import *
import numpy as np
import time
import csv_parse

def make_x(period_in_dt=21, x_scale=1, dt_test=.0001):
    times = x_scale * np.linspace(0, 1 + dt_test * period_in_dt, 1/dt_test) * Hz
    return times

def make_sine(period_in_dt=21, dt_test=.0001):
    # set up sine
    times = (1 / (dt_test * period_in_dt) ) * 2 * np.pi * np.linspace(0, 1 + dt_test * period_in_dt, 1/dt_test)
    arr = (10 * np.sin(times) + 1000) * Hz
    return arr

def make_snn_and_run_once(ts, lags=[2, 3, 5], duration=1*second, dt_ts=0.0001 * second):
    # constants, equations, detritus
    start_scope()
    print(ts)
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
            rates='ts(t - lags_ts(i * second) * {} * second)'.format(dt_ts),
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
    net = Network(ash, input, neurons, S, S2, sss)
    net.run(duration, report='text')
    #TODO: is this a use after free fuxie?
    return sss

def run_many_times(ts, aggregator, runs, lags=[2, 3, 5], duration=1*second, dt_ts=0.0001 * second):
    for i in range(runs):
        sss = make_snn_and_run_once(ts, lags, duration, dt_ts)
        aggregator(i, sss)
        print("Run {} done".format(i))

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

def lag_is_max(lags):
    lgd = {i: 0 for i in lags}
    def aggie(i, sss):
        it = -1
        for it in zip(*sss.w[0:len(lags)]):
            pass
        ml = lags[max(range(len(lags)), key=lambda x: it[x])]
        lgd[ml] += 1
    return aggie, lgd

def train_and_run(train_data, test_data, lags=[2, 3, 5], duration=1*second, dt_ts=0.0001*second, test_dur=None):
    sss = make_snn_and_run_once(train_data, lags, duration, dt_ts)
    lag_to_w = {l: sss.w[:][idx][-1] for idx, l in enumerate(lags)}
    print("Got weights", lag_to_w)
    lag_to_w = [lag_to_w[i] for i in lags]

    # brian detrius
    start_scope()
    N = 1000
    taum = 10*ms
    taue = 5*ms
    Ee = 0*mV
    vt = -54*mV
    vr = -60*mV
    El = -74*mV
    eqs_neurons = '''
    dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    '''
    lags_ts = TimedArray(lags, dt = 1 * second)
    ts = TimedArray(test_data, dt=dt_ts)
    input = PoissonGroup(len(lags),
            rates='ts(t - lags_ts(i * second) * {} * second)'.format(dt_ts),
            dt=0.0001 * second)
    neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                        method='euler',dt=0.0001 * second)
    S2 = Synapses(input, neurons,
                '''w : 1''',
                on_pre='''ge += w ''',
                )
    S2.connect()
    S2.w = lag_to_w

    if test_dur is None: test_dur = duration
    mon = SpikeMonitor(neurons, variables = ['v'])
    net = Network(input, neurons, S2, mon)
    net.run(test_dur, report='text')
    #TODO: is this too a use after free?
    return mon.values('t')[0]

def rms_error(spikes, observed, rate_est_window=1, dt_ts=0.0001 * second):
    observed = TimedArray(observed, dt=dt_ts) #easier to force brian to handle the dt bs.

    if len(spikes) < rate_est_window:
        return -1

    error = 0
    for i in range(rate_est_window, len(spikes) - rate_est_window):
        rate = (2 * rate_est_window + 1) / (spikes[i + rate_est_window] - spikes[i - rate_est_window])
        error += (rate - observed(spikes[i])) ** 2
    return np.sqrt(error/(len(spikes) - 2 * rate_est_window))

if __name__ == "__main__":
    daddy_bezos = csv_parse.returnNon2019Data('data/AMZN.csv') * Hz
    
    test = csv_parse.return2019Data('data/AMZN.csv') * Hz
    spoke = train_and_run(daddy_bezos, test)
    list(map(print,spoke))
    print(rms_error(spoke, test))
