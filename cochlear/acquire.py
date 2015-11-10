import collections
from threading import Thread

import numpy as np
from scipy import signal

from neurogen import block_definitions as blocks
from experiment.coroutine import coroutine, call, broadcast

from cochlear import nidaqmx as ni
from cochlear import calibration


@coroutine
def abr_reject(reject_threshold, fs, window, target_pass, target_fail):
    Wn = 0.2e3/fs, 10e3/fs
    b, a = signal.iirfilter(output='ba', N=1, Wn=Wn, btype='band',
                            ftype='butter')
    window_samples = int(window*fs)
    while True:
        data = (yield)[..., :window_samples]
        d = signal.filtfilt(b, a, data, axis=-1)
        if np.all(np.max(np.abs(d), axis=-1) < reject_threshold):
            target_pass.send(data)
        else:
            target_fail.send(data)


@coroutine
def extract_epochs(epoch_size, queue, buffer_size, target):
    buffer_size = int(buffer_size)
    data = (yield)
    buffer_shape = list(data.shape)
    buffer_shape[-1] = buffer_size
    ring_buffer = np.empty(buffer_shape, dtype=data.dtype)
    next_offset = None
    t0 = -buffer_size

    while True:
        # Newest data is always stored at the end of the ring buffer. To make
        # room, we discard samples from the beginning of the buffer.
        samples = data.shape[-1]
        a = ring_buffer[samples:]
        ring_buffer = np.concatenate((a, data), axis=-1)
        t0 += samples

        if next_offset is None and queue:
            next_offset = queue.popleft()

        while True:
            if next_offset < t0:
                raise SystemError, 'Epoch lost'
            elif (next_offset+epoch_size) > (t0+buffer_size):
                break
            else:
                i = next_offset-t0
                epoch = ring_buffer[..., i:i+epoch_size]
                target.send(epoch)
                if queue:
                    next_offset = queue.popleft()
                else:
                    next_offset = None
                    break

        data = (yield)


class ABRAcquisition(Thread):

    def __init__(self, frequencies, levels, calibration, duration=5e-3,
                 ramp_duration=0.5e-3, averages=512, window=8e-3,
                 reject_threshold=np.inf, repetition_rate=20, adc_fs=200e3,
                 dac_fs=200e3, stream_callback=None, epoch_callback=None):

        for k, v in locals().items():
            setattr(self, k, v)

        channel = blocks.Tone(name='tone') >> \
            blocks.Cos2Envelope(rise_time=ramp_duration, duration=duration) >> \
            ni.DAQmxChannel(calibration=calibration,
                            attenuator=ni.DAQmxBaseAttenuator())

        # Round up to nearest multiple of 2 since we need to do half at inverted
        # polarity
        pip_averages = int(np.ceil(averages/2.0))

        # Pipeline to group acquisitions into pairs of two (i.e. alternate
        # polarity tone-pips), reject the pair if either exceeds artifact
        # reject, and accumulate the specified averages.  When the specified
        # number of averages are acquired, the program exits.
        reject = abr_reject(reject_threshold, adc_fs, window,
                            call(self.process_valid_epochs),
                            call(self.process_invalid_epochs))
        pipeline = broadcast(call(self.process_stream), reject)
        input_samples = int(1.0/repetition_rate*adc_fs)
        iface_adc = ni.DAQmxInput(
            fs=adc_fs,
            input_line=ni.DAQmxDefaults.ERP_INPUT,
            callback_samples=input_samples,
            pipeline=pipeline,
            expected_range=10,
            start_trigger='ao/StartTrigger',
        )

        iface_dac = ni.QueuedDAQmxPlayer(
            output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
            duration=1.0/repetition_rate,
            buffer_size=0.5,
            monitor_interval=0.05,
        )
        iface_dac.add_channel(channel, name='primary')
        iface_dac.queue_init('Random')
        iface_dac.register('queue_pop', self.queue_pop_cb)

        token_value_map = {}
        token_acquired = {}
        waveforms = {}
        for phase in (0, np.pi):
            for level in levels:
                for frequency in frequencies:
                    values = {
                        'primary.tone.phase': phase,
                        'primary.tone.level': level,
                        'primary.tone.frequency': frequency,
                    }
                    uuid = iface_dac.queue_append(pip_averages, values=values)
                    token_value_map[(frequency, level, phase)] = uuid
                    token_acquired[uuid] = 0
                    waveforms[uuid] = []

        self.token_value_map = token_value_map
        self.token_sequence = collections.deque()
        self.token_acquired = token_acquired
        self.pip_averages = pip_averages
        self.waveforms = waveforms

        attens = iface_dac.get_best_overall_attenuations()
        iface_dac.set_attenuations(attens)
        self.iface_adc = iface_adc
        self.iface_dac = iface_dac

        super(ABRAcquisition, self).__init__()

    def run(self):
        self.iface_adc.start()
        self.iface_dac.play_queue(decrement=False)
        self.iface_adc.join()

    def process_valid_epochs(self, epochs):
        for epoch in epochs:
            try:
                token = self.token_sequence.popleft()
                print token['delay']
                print token['offset']
                uuid = token['uuid']
                self.iface_dac.current_queue.decrement_key(uuid)
                if self.token_acquired[uuid] < self.pip_averages:
                    self.waveforms[uuid].append(epoch)
                    self.token_acquired[uuid] += 1
                    if self.epoch_callback is not None:
                        self.epoch_callback()
            except KeyError:
                pass
            except IndexError:
                self.iface_adc.stop()
                self.iface_dac.stop()

    def process_invalid_epochs(self, epochs):
        for epoch in epochs:
            try:
                uuid = self.token_sequence.popleft()['uuid']
            except KeyError:
                pass
            except IndexError:
                self.iface_adc.stop()
                self.iface_dac.stop()

    def process_stream(self, stream):
        if self.stream_callback is not None:
            self.stream_callback(stream)

    def queue_pop_cb(self, event_type, event_data):
        self.token_sequence.append(event_data)

    def get_waveforms(self, frequency, level):
        w1 = self.waveforms[self.token_value_map[frequency, level, 0]]
        w2 = self.waveforms[self.token_value_map[frequency, level, np.pi]]
        trials = min(len(w1), len(w2))
        return np.concatenate((w1[:trials], w2[:trials]), axis=0)


def test_acquisition():
    # Connect ao0 to ai0 and ai1. The resulting plot should show noise, no tone
    # pips.
    import os.path
    from neurogen.calibration import InterpCalibration

    # Override standard ABR reject with specialized fucntion
    @coroutine
    def abr_reject(reject_threshold, fs, window, target_pass, target_fail):
        window_samples = int(window*fs)
        while True:
            data = (yield)[..., :window_samples]
            if np.random.random() >= 0.2:
                target_pass.send(data)
            else:
                target_fail.send(data)
    globals()['abr_reject'] = abr_reject

    mic_file = os.path.join('c:/data/cochlear/calibration',
                            '150807 - Golay calibration with 377C10.mic')
    input_calibration = InterpCalibration.from_mic_file(mic_file)
    input_calibration.set_fixed_gain(-40)
    frequencies = [2e3, 4e3, 8e3]
    levels = [60, 40, 20, 10]

    output_calibration = calibration.multitone_calibration(
        frequencies, input_calibration, gain=-40)

    acq = ABRAcquisition(frequencies, levels, output_calibration, averages=2)
    acq.start()
    acq.join()

    import pylab as pl
    for frequency in frequencies:
        for level in levels:
            w = acq.get_waveforms(frequency, level).mean(axis=0)[0]
            pl.plot(w, label='{} {}'.format(frequency, level))

    pl.legend()
    pl.show()


if __name__ == '__main__':
    test_acquisition()
