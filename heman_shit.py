user_input = input

from brian2 import *
import numpy as np
import time
import csv_parse
import itertools as it

def make_x(period_in_dt=21, x_scale=1, dt_test=.0001):
    times = x_scale * np.linspace(0, 1 + dt_test * period_in_dt, 1/dt_test) * Hz
    return times

def make_sine(period_in_dt=21, dt_test=.0001):
    # set up sine
    times = (1 / (dt_test * period_in_dt) ) * 2 * np.pi * np.linspace(0, 1 + dt_test * period_in_dt, 1/dt_test)
    arr = (10 * np.sin(times) + 1000) * Hz
    return arr

def make_snn_and_run_once(ts, lags=[2, 3, 5], duration=None, dt_ts=0.0001 * second, normalization=None):
    # constants, equations, detritus
    start_scope()
    if normalization is None: normalization = max(ts)
    if duration is None: duration = len(ts)
    ts = TimedArray(ts, dt=dt_ts * normalization)
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
    input_neur = PoissonGroup(len(lags),
            rates='ts(t - lags_ts(i * second) * {} * second)'.format(dt_ts),
            dt=0.0001 * second)
    neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                        method='euler',dt=0.0001 * second)

    # synapses
    S = Synapses(input_neur, neurons,
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
    S.w = 1 #.01
    S2 = Synapses(ash, neurons,
                '''w : 1''',
                on_pre='''ge += w ''',
                )
    S2.connect()
    S2.w = 1 #0.01

    # Monitors
    # sss = StateMonitor(S, variables=['w'], record=range(10000), dt=dt_ts)
    # mon = StateMonitor(neurons, variables = ['v'],record=range(10000), dt=0.0001 * second )
    # mon = PopulationRateMonitor(neurons)

    # Run and record
    # net = Network(ash, input_neur, neurons, S, S2, sss)
    net = Network(ash, input_neur, neurons, S, S2)
    net.run(duration * normalization * dt_ts, report='text')
    #TODO: is this a use after free fuxie?
    # d = list(zip(mon.t, mon.smooth_rate(window="flat", width=normalization * dt_ts * second * second)))
    # list(map(print, d))
    # plot([i[0] for i in d], [i[1] for i in d])
    # show()
    return S.w

#WARNING: deprecated
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

def train_and_run(train_data, test_data, lags=[2, 3, 5], dt_ts=0.0001*second,
        rate_est_window=None):
    #TODO: normalize: max(ts) is OK but not enough for increasing series (esp given cross validation)
    # this feels like cheating - I'm just checking if it works
    normie = max(max(train_data), max(test_data)) * second
    if rate_est_window is None: rate_est_window = normie
    duration = len(test_data)

    print("Expected durations: training {} testing {}".format(dt_ts * normie * len(train_data), dt_ts * normie * len(test_data)))
    if user_input("Continue?") == "no":
        import sys
        sys.exit()

    sss = make_snn_and_run_once(train_data, lags, dt_ts=dt_ts, normalization=normie)
    print("Got weights", sss)

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
    ts = TimedArray(test_data, dt=dt_ts * normie)
    input_neur = PoissonGroup(len(lags),
            rates='ts(t - lags_ts(i * second) * {} * second)'.format(dt_ts),
            dt=0.0001 * second)
    neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                        method='euler',dt=0.0001 * second)
    S2 = Synapses(input_neur, neurons,
                '''w : 1''',
                on_pre='''ge += w ''',
                )
    S2.connect()
    S2.w = sss

    mon = PopulationRateMonitor(neurons)
    net = Network(input_neur, neurons, S2, mon)
    net.run(dt_ts * normie * duration, report='text')
    #TODO: is this too a use after free? - consume iter to avoid
    return list(zip(mon.t, mon.smooth_rate(window='flat', width=rate_est_window * dt_ts)))

def merge_lists_by(big, sub, merger, dt):
    """
    Types:
        big: [(second, Hz)]
        sub: [(second, Hz)]
        merger: (second, Hz) -> (second, Hz) -> a
        dt: second
        return type: [a]

    Merge sort's merge, but time tolerance is dt (for equality checking).
    """
    sub_i = 0
    big_i = 0
    while big_i < len(big) and sub_i < len(sub):
        if abs(big[big_i][0] - sub[sub_i][0]) < dt:
            yield merger(big[big_i], sub[sub_i])
            big_i += 1
            sub_i += 1
        elif big[big_i][0] > sub[sub_i][0]:
            yield merger(None, sub[sub_i])
            sub_i += 1
        else:
            yield merger(big[big_i], None)
            big_i += 1

    yield from map(lambda b: merger(b, None), big[big_i:])
    yield from map(lambda s: merger(None, s), sub[sub_i:])

def rms_error(spikes, observed, dt_ts=0.0001 * second):
    def term_error(exp, obs):
        if exp is None:
            return obs[1] ** 2
        elif obs is None:
            return exp[1] ** 2
        else:
            return (exp[1] - obs[1]) ** 2

    norman = int(max(observed)) + 1
    timings = np.linspace(0, len(observed) * norman * dt_ts, len(observed) * norman)
    observed = list(o for l in map(lambda i: it.repeat(i, norman), observed) for o in l)
    error = sum(merge_lists_by(list(zip(timings, observed)), spikes, term_error, dt_ts/2))
    return np.sqrt(error/len(observed))

def plot_exp_vs_obs(spikes, observed, dt_ts=0.0001 * second):
    def print_and_grab_tuples(exp, obs):
        if exp is None:
            print("t = {}, exp = 0, obs = {}".format(*obs))
            return (obs[0], 0), obs
        if obs is None:
            print("t = {}, exp = {}, obs = 0".format(*exp))
            return exp, (exp[0], 0)

        print("t = {}, exp = {}, obs = {}".format(obs[0], exp[1], obs[1]))
        return exp, obs

    norman = int(max(observed)) + 1
    timings = np.linspace(0, len(observed) * norman * dt_ts, len(observed) * norman)
    observed = (o for l in map(lambda i: it.repeat(i, norman), observed) for o in l)
    lot = list(merge_lists_by(list(zip(timings, observed)), spikes, print_and_grab_tuples, dt_ts/2))
    exps = [i[0][1] for i in lot]
    obss = [i[1][1] for i in lot]
    plot(exps)
    plot(obss)
    show()

if __name__ == "__main__":
    daddy_bezos = csv_parse.returnNon2019Data('data/AMZN.csv') * Hz
    test = csv_parse.return2019Data('data/AMZN.csv') * Hz

    test_dt = 0.0001 * second
    spoke = train_and_run(daddy_bezos, test, dt_ts=test_dt)

    print('RMS ERROR', rms_error(spoke, test, test_dt))
    plot_exp_vs_obs(spoke, test, test_dt)
