user_input = input

import itertools as it

from brian2 import *
import numpy as np
import time
import csv_parse

FILE = "data/IBM.csv"

def make_snn_and_run_once(ts, numNeurons, runs, dt_ts=0.0001 * second):
    # constants, equations, detritus
    start_scope()
    duration = len(ts)

    idxs, ts2 = csv_parse.buildInputArray(numNeurons,ts, repeats=runs)
    input_neur = SpikeGeneratorGroup(numNeurons, idxs, ts2*dt_ts)
    #5*dt_ts is the lag
    idxs, ts = csv_parse.buildInputArray(numNeurons, ts, 5 * dt_ts * Hz, repeats=runs)
    ash_excite = SpikeGeneratorGroup(numNeurons, idxs, ts * dt_ts)
    ash_inhib = SpikeGeneratorGroup(numNeurons, idxs, ts * dt_ts)

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
    mon = SpikeMonitor(neurons)
    # Run and record
    net = Network(input_neur, neurons, S, mon, ash_excite, S2, ash_inhib, S3)
    for j in range(runs):
        print("training iter ", j)
        net.run(duration  * dt_ts * (j + 1), report='text')

    spoke = mon.spike_trains()

    print("GAY",mon.spike_trains())

    # d = list(zip(mon.t, mon.smooth_rate(window="flat", width=normalization * dt_ts * second * second)))
    # list(map(print, d))
    # plot([i[0] for i in d], [i[1] for i in d])
    # show()
    return S.w

def train_and_run(train_data, test_data, numNeurons, runs, dt_ts=0.0001*second):
    #TODO: normalize: max(ts) is OK but not enough for increasing series (esp given cross validation)
    # this feels like c
    normie = max(max(train_data), max(test_data)) * second
    duration = len(test_data)

    sss = make_snn_and_run_once(train_data, numNeurons, runs, dt_ts=dt_ts)
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
    idxs, ts = csv_parse.buildInputArray(numNeurons, test_data, repeats=runs)
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
    for t in range(runs):
        print("testing iter", t)
        net.run(dt_ts * duration * (t + 1), report='text')
    #TODO: is this too a use after free? - consume iter to avoid
    #return list(zip(mon.t, mon.smooth_rate(window='flat', width=rate_est_window * dt_ts)))
    spike_trains = mon.spike_trains()
    print('RETARDED', spike_trains)
    return spike_trains

def plot_data_and_spikes(data, spike_mon, test_dt, min_run=0, unique=False):
    """
        Expects data to be scaled to the buckets that the spike monitor, spike_mon
        monitored.

        The dt - test_dt - is to scale spike times to indices in the data.

        min_run lets you lower bound the run you wish to see. It's assumed that
        a run is of duration len(data) * test_dt.

        unique is set to plot only differing x values (so if all neurons spike at
        the same time, this will effectively pick a random representative).k
    """
    y_list, x_list = [], []
    uniq = dict()
    for neuron in spike_mon:
        for time in spike_mon[neuron]:
            if time >= min_run * len(data) * test_dt:
                x_list.append(time / test_dt * second)
                y_list.append(neuron * Hz)
            if unique and int(time / test_dt) not in uniq and time >= min_run * len(data) * test_dt:
                uniq[int(time / test_dt) % len(data)] = neuron * second

    if unique:
        scatter(uniq.keys(), uniq.values(), color="red")
    else:
        scatter(x_list, y_list, color="red")
    plot(csv_parse.buildInputArray(100, test)[0], color="blue")
    show()

if __name__ == "__main__":
    daddy_bezos = csv_parse.return2018Data(FILE) * Hz
    test = csv_parse.return2019Data(FILE) * Hz
    # test = np.fromiter(it.repeat(min(test), test.shape[0] * 3), int) * Hz

    test_dt = 0.0001 * second
    buckets = 100
    runs = 100
    #spoke = list(train_and_run(daddy_bezos, test, [1], dt_ts=test_dt))
    spoke = train_and_run(daddy_bezos, test, buckets, runs, dt_ts=test_dt)
    plot_data_and_spikes(test, spoke, test_dt, runs - 1, True)
#    print(rms_error(spoke, test, test_dt))
#    plot_exp_vs_obs(spoke, test, test_dt)
