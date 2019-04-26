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

def make_snn_and_run_once(ts, lags=[2, 3, 5], duration=1*second, dt_ts=0.0001 * second, normalization=None):
    # constants, equations, detritus
    start_scope()
    if normalization is None: normalization = max(ts)
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
    sss = StateMonitor(S, variables=['w'], record=range(10000), dt=dt_ts)
    # mon = StateMonitor(neurons, variables = ['v'],record=range(10000), dt=0.0001 * second )
    mon = PopulationRateMonitor(neurons)

    # Run and record
    net = Network(ash, input, neurons, S, S2, sss, mon)
    net.run(duration, report='text')
    #TODO: is this a use after free fuxie?
    list(map(print, zip(mon.t, mon.smooth_rate(window="flat", width=1*second))))
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

def train_and_run(train_data, test_data, lags=[2, 3, 5], duration=1*second, dt_ts=0.0001*second,
        test_dur=None, rate_est_window=1):
    #TODO: normalize: max(ts) is OK but not enough for increasing series (esp given cross validation)
    # this feels like cheating - I'm just checking if it works
    normie = max(max(train_data), max(test_data)) * second

    sss = make_snn_and_run_once(train_data, lags, duration, dt_ts, normie)
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
    ts = TimedArray(test_data, dt=dt_ts * normie)
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
    mon = PopulationRateMonitor(neurons)
    net = Network(input, neurons, S2, mon)
    net.run(test_dur, report='text')
    #TODO: is this too a use after free? - consume iter to avoid
    return list(zip(mon.t, mon.smooth_rate(window='flat', width=rate_est_window * second)))

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

    timings = np.linspace(0, len(observed)*dt_ts, len(observed))
    error = sum(merge_lists_by(list(zip(timings, observed)), spikes, term_error, dt_ts/2))
    #Expectation: there won't be spikes at times that aren't observations.
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

    timings = np.linspace(0, len(observed)*dt_ts, len(observed))
    lot = list(merge_lists_by(list(zip(timings, observed)), spikes, print_and_grab_tuples, dt_ts/2))
    exps = [i[0] for i in lot]
    obss = [i[1] for i in lot]
    plot(exps)
    plot(obss)
    show()

if __name__ == "__main__":
    daddy_bezos = csv_parse.returnNon2019Data('data/AMZN.csv') * Hz
    test = csv_parse.return2019Data('data/AMZN.csv') * Hz

    bezos_bucks = len(daddy_bezos)
    test_bucks = len(test)
    test_dt = 0.0001 * second
    spoke = train_and_run(daddy_bezos, test, duration=bezos_bucks * test_dt, test_dur=test_bucks * test_dt, dt_ts=test_dt, rate_est_window=1)

    print(rms_error(spoke, test, test_dt))
    plot_exp_vs_obs(spoke, test, test_dt)
