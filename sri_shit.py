user_input = input

import itertools as it

from brian2 import *
import numpy as np
import time
import csv_parse

FILE = "data/IBM.csv"

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
    if duration is None: duration = len(ts)

    numNeurons = 100 # csv_parse.getMinMaxDiff(FILE)
    idxs, ts2 = csv_parse.buildInputArray(numNeurons,ts, repeats=100)
    test = idxs
    input_neur = SpikeGeneratorGroup(numNeurons, idxs, ts2*dt_ts)
    #5*dt_ts is the lag
    idxs, ts = csv_parse.buildInputArray(numNeurons, ts, 5 * dt_ts * Hz, repeats=100)
    ash_excite = SpikeGeneratorGroup(numNeurons, idxs, ts * dt_ts)
    ash_inhib = SpikeGeneratorGroup(numNeurons, idxs, ts * dt_ts)

    N = 1000
    taupre = 20*ms
    taupost = taupre
    taue = 1/0.9 *ms
    gmax = 10
    dApre = .01
    dApost = -dApre * taupre / taupost * 1.05
    dApost *= gmax
    dApre *= gmax
    ged = 1
    a = 0.02/ms
    b = 0.2/ms
    c = -65*mV # resting potential
    d = 8*mV/ms

    reset ='''
        v = c
        u += d
        '''

    eqs = '''
        dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u + I : volt
        du/dt = a*(b*v-u) : volt/second
        dI/dt = -I / taue : volt/second
        '''
        # old: dI/dt = -I / taue : volt/second

    neurons = NeuronGroup(numNeurons,eqs, threshold='v>30*mV', reset=reset,
                        method='euler',dt=dt_ts)

    # synapses
    S = Synapses(input_neur, neurons,
                '''w : 1
                    dApre/dt = -Apre / taupre : 1 (event-driven)
                    dApost/dt = -Apost / taupost : 1 (event-driven)''',
                on_pre='''I  +=w / radian * volt/second
                        Apre += dApre
                        w = clip(w + Apost, 0, gmax)''',
                on_post='''Apost += dApost
                        w = clip(w + Apre, 0, gmax)''',
                )
    S.connect()
    # S.w = np.random.rand(numNeurons ** 2)
    S.w = 6
    S2 = Synapses(ash_excite, neurons,
                '''w : 1''',
                on_pre='''I  +=w / radian * volt/second ''',
                )
    S2.connect('i==j')
    S2.w = 6
    S3 = Synapses(ash_inhib, neurons,
                '''w : 1''',
                on_pre='''I  +=w / radian * volt/second ''',
                )
    S3.connect('i!=j')
    S3.w = -5

    # Monitors
    # sss = StateMonitor(S, variables=['w'], record=range(10000), dt=dt_ts)
    # mon = StateMonitor(neurons, variables = ['v'],record=range(10000), dt=0.0001 * second )
    # mon = PopulationRateMonitor(neurons)
    mon = SpikeMonitor(neurons)
    # Run and record
    # net = Network(ash, input_neur, neurons, S, S2, sss)
    net = Network(input_neur, neurons, S, mon, ash_excite, S2, ash_inhib, S3)
    for j in range(100):
        print("iter ", j)
        net.run(duration  * dt_ts * (j + 1), report='text')

    spoke = mon.spike_trains()
    y_list, x_list = [], []
    uniq_pts = dict()
    for neuron in spoke:
        for time in spoke[neuron]:
            x_list.append(time * 10 * 1000)
            y_list.append(min_stock + neuron * Hz)
    scatter(x_list, y_list, color="red")
    plot(test, color="blue")
    show()

    print("GAY",mon.spike_trains())

    # d = list(zip(mon.t, mon.smooth_rate(window="flat", width=normalization * dt_ts * second * second)))
    # list(map(print, d))
    # plot([i[0] for i in d], [i[1] for i in d])
    # show()
    return S.w

def train_and_run(train_data, test_data, lags=[2, 3, 5], dt_ts=0.0001*second,
        rate_est_window=None):
    #TODO: normalize: max(ts) is OK but not enough for increasing series (esp given cross validation)
    # this feels like c
    normie = max(max(train_data), max(test_data)) * second
    if rate_est_window is None: rate_est_window = normie
    duration = len(test_data)

    sss = make_snn_and_run_once(train_data, lags, dt_ts=dt_ts, normalization=normie)
    print("Got weights", sss)

    # brian detrius
    start_scope()
    taue = 1/0.9 *ms
    a = 0.02/ms
    b = 0.2/ms
    c = -65*mV # resting potential
    d = 8*mV/ms

    reset ='''
        v = c
        u += d
        '''

    eqs = '''
        dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u + I : volt
        du/dt = a*(b*v-u) : volt/second
        dI/dt = -I / taue : volt/second
        '''
    numNeurons = 100 #csv_parse.getMinMaxDiff(FILE)
    min_stock = min(test_data)
    idxs, ts = csv_parse.buildInputArray(numNeurons, test_data, repeats=100)
    input_neur = SpikeGeneratorGroup(numNeurons, idxs, ts*dt_ts)

    neurons = NeuronGroup(numNeurons, eqs, threshold='v>30*mV', reset=reset,
                        method='euler',dt=dt_ts)

    S2 = Synapses(input_neur, neurons,
                '''w : 1''',
                on_pre='''I += w / radian * volt/second ''',
                )
    S2.connect()
    S2.w = sss

    mon = SpikeMonitor(neurons)
    net = Network(input_neur, neurons, S2, mon)
    for t in range(100):
        print("iter", t)
        net.run(dt_ts * duration * (t + 1), report='text')
    #TODO: is this too a use after free? - consume iter to avoid
    #return list(zip(mon.t, mon.smooth_rate(window='flat', width=rate_est_window * dt_ts)))
    spike_trains = mon.spike_trains()
    print('RETARDED', spike_trains)
    return spike_trains
#    print('RET', duration, duration * dt_ts, np.linspace(0, duration * dt_ts, duration))
#    for t in np.linspace(0, duration * dt_ts, duration):
#        whomst, denom = 0, 0
#        for neuron in spike_trains:
#            closest_idx = min(range(len(spike_trains[neuron])), key=lambda i: abs(spike_trains[neuron][i] - t))
#            if abs(spike_trains[neuron][closest_idx] - t) < 5 * dt_ts:
#                neuron_dec = min_stock + neuron *Hz
#                whomst += neuron_dec
#                denom += 1
#            spike_trains[neuron][closest_idx] = -12 * second
#        avg = 0
#        if denom != 0:
#            avg = whomst/denom
#        print("WHOMST",whomst)
#        print("DENOM",denom)
#        yield (t , avg )

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
    exps = [i[0][1] for i in lot]
    obss = [i[1][1] for i in lot]
    plot(exps, color="blue")
    plot(obss, color="red")
    show()

if __name__ == "__main__":
    daddy_bezos = csv_parse.return2018Data(FILE) * Hz
    test = csv_parse.return2019Data(FILE) * Hz
    # test = np.fromiter(it.repeat(min(test), test.shape[0] * 3), int) * Hz

    test_dt = 0.0001 * second
    #spoke = list(train_and_run(daddy_bezos, test, [1], dt_ts=test_dt))
    min_stock = min(min(test), min(daddy_bezos))
    spoke = train_and_run(daddy_bezos, test, [1], dt_ts=test_dt)
    y_list, x_list = [], []
    uniq = dict()
    for neuron in spoke:
        for time in spoke[neuron]:
            x_list.append(time * 10 * 1000)
            y_list.append(min_stock + neuron * Hz)
            if (time * 10000 * Hz) not in uniq and time >= 99 * len(test) * test_dt:
                uniq[time * 10000 * Hz] = min_stock + neuron * Hz

    print("ore wa mou plotto, ikimashou")
    scatter(x_list, y_list, color="red")
    plot(csv_parse.buildInputArray(100, test, repeats=100)[0], color="blue")
    show()
    scatter(uniq.keys(), uniq.values(), color="red")
    plot(csv_parse.buildInputArray(100, test, repeats=100)[0], color="blue")
    show()
#    print(rms_error(spoke, test, test_dt))
#    plot_exp_vs_obs(spoke, test, test_dt)
