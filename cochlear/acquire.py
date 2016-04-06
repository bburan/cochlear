import logging
log = logging.getLogger(__name__)

import collections
from threading import Thread, Event

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
        ring_buffer[..., :-samples] = ring_buffer[..., samples:]
        ring_buffer[..., -samples:] = data
        t0 += samples

        while True:
            if (next_offset is None) and len(queue) > 0:
                next_offset = queue.popleft()
            elif next_offset is None:
                break
            elif next_offset < t0:
                raise SystemError('Epoch lost')
            elif (next_offset+epoch_size) > (t0+buffer_size):
                break
            else:
                i = next_offset-t0
                epoch = ring_buffer[..., i:i+epoch_size].copy()
                target.send(epoch)
                next_offset = None

        data = (yield)


class ABRAcquisition(Thread):

    def __init__(self, frequencies, levels, calibration, duration=5e-3,
                 ramp_duration=0.5e-3, pip_averages=512, window=8.5e-3,
                 reject_threshold=np.inf, repetition_rate=20, adc_fs=200e3,
                 dac_fs=200e3, samples_acquired_callback=None,
                 valid_epoch_callback=None, invalid_epoch_callback=None,
                 done_callback=None):

        log.debug('Initializing ABR acquisition')
        for k, v in locals().items():
            setattr(self, k, v)

        channel = blocks.Tone(name='tone') >> \
            blocks.Cos2Envelope(rise_time=ramp_duration, duration=duration) >> \
            ni.DAQmxChannel(calibration=calibration,
                            attenuator=ni.DAQmxBaseAttenuator())

        input_samples = int(1.0/repetition_rate*adc_fs)*5

        reject = abr_reject(reject_threshold, adc_fs, window,
                            call(self.process_valid_epochs),
                            call(self.process_invalid_epochs))
        self.token_offsets = collections.deque()
        extract = extract_epochs(input_samples, self.token_offsets, dac_fs*10,
                                 reject)
        pipeline = broadcast(call(self.process_samples), extract)

        iface_adc = ni.DAQmxInput(
            fs=adc_fs,
            input_line=ni.DAQmxDefaults.ERP_INPUT,
            pipeline=pipeline,
            expected_range=1,
            start_trigger='ao/StartTrigger',
            record_mode=ni.DAQmxInput.DIFF,
            callback_samples=input_samples,
            done_callback=self.stop,
        )

        iface_dac = ni.QueuedDAQmxPlayer(
            fs=dac_fs,
            output_line=ni.DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
            duration=1.0/repetition_rate,
            buffer_size=0.5,
            monitor_interval=0.05,

        )
        iface_dac.add_channel(channel, name='primary')
        iface_dac.queue_init('Interleaved FIFO')
        iface_dac.register('queue_pop', self.queue_pop_cb)

        self.value_to_uuid = {}
        self.uuid_to_value = {}
        self.current_averages = {}
        self.waveforms = {}
        for phase in (0, np.pi):
            for level in levels:
                for frequency in frequencies:
                    values = {
                        'primary.tone.phase': phase,
                        'primary.tone.level': level,
                        'primary.tone.frequency': frequency,
                    }
                    uuid = iface_dac.queue_append(pip_averages, values=values)
                    value = (frequency, level, phase)
                    self.value_to_uuid[value] = uuid
                    self.uuid_to_value[uuid] = value
                    self.current_averages[uuid] = 0
                    self.waveforms[uuid] = []

        self.token_sequence = collections.deque()
        self.pip_averages = pip_averages

        attens = iface_dac.get_best_overall_attenuations()
        iface_dac.set_attenuations(attens)
        self.iface_adc = iface_adc
        self.iface_dac = iface_dac

        self.adc_dac_ratio = adc_fs/dac_fs

        self._stop = Event()
        self.state = 'initialized'
        super(ABRAcquisition, self).__init__()

    def run(self):
        self.iface_adc.start()
        self.iface_dac.play_queue(decrement=False)
        self.state = 'running'
        self.iface_adc.join()

    def request_stop(self):
        log.debug('Stop requested')
        self._stop.set()

    def stop(self):
        log.debug('Stopping acquisition')
        self._stop.clear()
        self.iface_adc.clear()
        self.iface_dac.clear()
        if self.done_callback is not None:
            log.debug('Notifying callback')
            self.done_callback(self.state)

    def process_valid_epochs(self, epochs):
        if self.state != 'running':
            return
        for epoch in epochs:
            uuid = self.token_sequence.popleft()['uuid']
            if self.current_averages[uuid] < self.pip_averages:
                self.waveforms[uuid].append(epoch)
                if self.valid_epoch_callback is not None:
                    value = list(self.uuid_to_value[uuid])
                    value.append(self.current_averages[uuid])
                    self.valid_epoch_callback(*value, epoch=epoch)
                self.current_averages[uuid] += 1
        self.check_status()

    def check_status(self):
        # If stop has not been requested, then check to see if we've acquired
        # all the epochs we need.
        if self._stop.isSet():
            log.debug('Stop requested')
            self.state = 'aborted'
            self.stop()
        else:
            for v in self.current_averages.values():
                if v < self.pip_averages:
                    return
            self.state = 'complete'
            self.stop()

    def process_invalid_epochs(self, epochs):
        if self.state != 'running':
            return
        for epoch in epochs:
            uuid = self.token_sequence.popleft()['uuid']
            if self.invalid_epoch_callback is not None:
                value = self.uuid_to_value[uuid]
                self.invalid_epoch_callback(*value, epoch=epoch)
        if self._stop.isSet() or self.iface_dac.current_queue.buffer_empty():
            self.iface_adc.stop()
            self.iface_dac.stop()

    def process_samples(self, samples):
        if self.state != 'running':
            return
        if self.samples_acquired_callback is not None:
            self.samples_acquired_callback(samples)

    def queue_pop_cb(self, event_type, event_data):
        self.token_sequence.append(event_data)
        offset = int(event_data['offset']*self.adc_dac_ratio)
        self.token_offsets.append(offset)

    def get_waveforms(self, frequency, level):
        w1 = self.waveforms[self.value_to_uuid[frequency, level, 0]]
        w2 = self.waveforms[self.value_to_uuid[frequency, level, np.pi]]
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
    frequencies = [2e3, 8e3]
    levels = [60, 40]

    def samples_acquired_cb(samples):
        print samples.shape

    def valid_epoch_cb(frequency, level, phase, presentation, epoch):
        print presentation, frequency, level, phase, epoch.shape

    output_calibration = calibration.multitone_calibration(
        frequencies, input_calibration, gain=-40)

    acq = ABRAcquisition(frequencies, levels, output_calibration,
                         pip_averages=2,
                         samples_acquired_callback=samples_acquired_cb,
                         valid_epoch_callback=valid_epoch_cb)
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
