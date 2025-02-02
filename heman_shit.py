user_input = input

import itertools as it

from brian2 import *
import numpy as np
import time
import csv_parse

PICKLE_JAR = "snns.brian"

FILE = "data/IBM.csv"

def make_snn_and_run_once(ts, numNeurons, runs, dt_ts=0.0001 * second, use_weights=None, save_as=None):
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
    if use_weights is not None:
        net.restore('training', use_weights)
        return S.w

    for j in range(runs):
        print("training iter ", j)
        net.run(duration  * dt_ts * (j + 1), report='text')

    if save_as is not None:
        net.store('training', save_as)

    spoke = mon.spike_trains()
    print("GAY",mon.spike_trains())

    # d = list(zip(mon.t, mon.smooth_rate(window="flat", width=normalization * dt_ts * second * second)))
    # list(map(print, d))
    # plot([i[0] for i in d], [i[1] for i in d])
    # show()
    return S.w

def train_and_run(train_data, test_data, numNeurons, runs, dt_ts=0.0001*second, use_weights=None, save_as=None):
    duration = len(test_data)

    sss = make_snn_and_run_once(train_data, numNeurons, runs, dt_ts=dt_ts, use_weights=use_weights, save_as=save_as)
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
    if use_weights is not None:
        net.restore('testing', use_weights)
        return mon.spike_trains()

    for t in range(runs):
        print("testing iter", t)
        net.run(dt_ts * duration * (t + 1), report='text')

    if save_as is not None:
        net.store('testing', save_as)

    spike_trains = mon.spike_trains()
    print('RETARDED', spike_trains)

    return spike_trains

def plot_data_and_spikes(data, spike_mon, test_dt, min_run=0, unique=False, buckets=100):
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
            if time >= min_run * test_dt:
                x_list.append(int((time + 5 * test_dt) / test_dt) % len(data))
                y_list.append(neuron * Hz)
            if unique and int((time + 5 * test_dt) / test_dt) not in uniq and time >= min_run * test_dt:
                uniq[int((time + 5 * test_dt) / test_dt) % len(data)] = neuron * second

    rms = 0
    mi = min(test)
    ma = max(test)
    for time in uniq:
        actual = int((test[time] - mi)/(ma - mi) * buckets)
        pred = uniq[time] * Hz
        rms += (actual - pred) ** 2

    if unique:
        scatter(uniq.keys(), uniq.values(), color="red")
    else:
        scatter(x_list, y_list, color="red")
    plot(csv_parse.buildInputArray(100, test)[0], color="blue")
    show()
    return np.sqrt(rms/len(data))

if __name__ == "__main__":
    from sys import argv
    train = "train" in argv
    to_test = train or ("test" in argv)
    to_plot = "no-plot" not in argv

    test_dt = 0.0001 * second
    buckets = 100
    runs = 100

    daddy_bezos = csv_parse.return2018Data(FILE) * Hz
    test = csv_parse.return2019Data(FILE) * Hz

    if train:
        spoke = train_and_run(daddy_bezos, test, buckets, runs, dt_ts=test_dt, save_as=PICKLE_JAR)
    elif to_test or to_plot:
        spoke = train_and_run(daddy_bezos, test, buckets, runs, dt_ts=test_dt, use_weights=PICKLE_JAR, save_as=PICKLE_JAR)
    if to_plot:
        # plot_data_and_spikes(test, spoke, test_dt, runs - 1, False)
        d = len(test)
        test = [o for l in it.repeat(test, runs//32) for o in l]
        print(plot_data_and_spikes(test, spoke, test_dt, 31 * runs//32 * d, unique=True))
