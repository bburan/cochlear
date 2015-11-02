'''
Classes for configuring and recording using NIDAQmx compatible devices
'''

from __future__ import division

import ctypes
import unittest
import time
import importlib

import PyDAQmx as ni
import numpy as np

from neurogen.channel import Channel
from neurogen.player import ContinuousPlayer, QueuedPlayer
from neurogen.blocks import ScalarInput
from neurogen import prepare_for_write

import threading

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

################################################################################
# Current configuration of niDAQmx hardware
################################################################################
class DAQmxDefaults(object):
    '''
    Define defaults for configuring the niDAQmx tasks.  Lines are typically
    hard-wired, so this is where you will configure the lines used for your
    specific hardware.
    '''
    DEV = 'PXI4461'

    MIC_INPUT = '/{}/ai0'.format(DEV)
    REF_MIC_INPUT = '/{}/ai1'.format(DEV)
    #ERP_INPUT = '/{}/ai1'.format(DEV)
    ERP_INPUT = MIC_INPUT
    AI_RANGE = 10

    AO_RANGE = np.sqrt(2)
    SPEAKER_OUTPUTS = (
        '/{}/ao0'.format(DEV),
        '/{}/ao1'.format(DEV),
    )
    PRIMARY_SPEAKER_OUTPUT = SPEAKER_OUTPUTS[0]

    AI_FS = 200e3
    AO_FS = 200e3

    ATTEN_CLASS = 'cochlear.nidaqmx.DAQmxBaseAttenuator'


################################################################################
# Utility functions
################################################################################
def get_bits(word, n, order='big-endian'):
    bitword = [(word >> i) & 1 for i in range(n)]
    if order == 'big-endian':
        return bitword[::-1]
    elif order == 'little-endian':
        return bitword
    else:
        raise ValueError('Byte order {} not recognized'.format(order))


################################################################################
# Functions for configuring common niDAQmx tasks
################################################################################
def create_task(name=None):
    '''
    Create niDAQmx task

    Parameters
    ----------
    name : {None, str}
        Task name (optional)

    Returns
    -------
    task : ctypes pointer
        Pointer to niDAQmx task
    '''
    if name is None:
        name = ''
    task = ni.TaskHandle(0)
    ni.DAQmxCreateTask(name, ctypes.byref(task))
    return task


def create_event_timer(trigger, fs, counter='/Dev1/Ctr0',
                       clock='ao/SampleClock', task=None):
    '''
    Create timer to report event timestamps

    Parameters
    ----------
    trigger : str
        Line to monitor for trigger
    counter : str
        Which counter channel to use
    clock : str
        Timebase for counter.  The value read from the counter will be in
        units of the specified clock.
    task : {None, niDAQmx task}
        If None, a new task will be created.

    Returns
    -------
    task : niDAQmx task
        Configured and committed task
    '''
    if task is None:
        task = create_task()
    ni.DAQmxCreateCICountEdgesChan(task, counter, '', ni.DAQmx_Val_Rising, 0,
                                   ni.DAQmx_Val_CountUp)
    ni.DAQmxSetCICountEdgesTerm(task, counter, clock)
    ni.DAQmxCfgSampClkTiming(task, trigger, fs, ni.DAQmx_Val_Rising,
                             ni.DAQmx_Val_FiniteSamps, 500)
    ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)
    return task


def create_ai(ai, fs, expected_range=10, callback=None, callback_samples=None,
              sample_clock='', start_trigger=None,
              record_mode=ni.DAQmx_Val_Diff):
    '''
    Parameters
    ----------
    ai : str
        Analog input line(s) to acquire in this task (e.g., /Dev1/ai0)
    expected_range : float
        Range of signal (in volts).
    callback : callable
        Function to call with acquired data
    callback_samples : int
    '''
    task = create_task()
    vlb, vub = -expected_range, expected_range
    ni.DAQmxCreateAIVoltageChan(task, ai, '', record_mode, vlb, vub,
                                ni.DAQmx_Val_Volts, '')

    ni.DAQmxCfgSampClkTiming(task, sample_clock, fs, ni.DAQmx_Val_Rising,
                             ni.DAQmx_Val_ContSamps, 100000)

    # Hint, this can be tied to ao/StartTrigger to start as soon as the analog
    # output task begins.
    if start_trigger is not None:
        ni.DAQmxCfgDigEdgeStartTrig(task, start_trigger, ni.DAQmx_Val_Rising)

    # Store a reference to the callback pointers on the task object itself. This
    # prevents the callback pointers from getting garbage-collected. If they get
    # garbage-collected, then the callback no longer exists!
    if callback is not None:
        task._cb_ptr = create_everynsamples_callback(callback, callback_samples,
                                                     task)
        ni.DAQmxCfgInputBuffer(task, callback_samples*100)
    ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)

    # Log some properties
    result = ctypes.c_double()
    ni.DAQmxGetAIFilterDelay(task, ai, result)
    log.debug('AI filter delay for %s is %.2f usec', ai, result.value*1e6)
    log.debug('AI filter delay for %s is %.2f samples', ai, result.value*fs)

    result = ctypes.c_double()
    ni.DAQmxGetAIGain(task, ai, result)
    log.debug('AI gain for %s is %.2f', ai, result.value)

    return task


def create_ao(ao, fs, expected_range=DAQmxDefaults.AO_RANGE, total_samples=None,
              buffer_size=None, callback=None, callback_samples=None,
              done_callback=None, start_trigger=None, sample_clock=''):

    task = create_task()

    # Setup analog output and prevent playout of data that has already been
    # played.
    ni.DAQmxCreateAOVoltageChan(task, ao, '', -expected_range, expected_range,
                                ni.DAQmx_Val_Volts, '')
    ni.DAQmxSetWriteRegenMode(task, ni.DAQmx_Val_DoNotAllowRegen)

    if start_trigger is not None:
        ni.DAQmxCfgDigEdgeStartTrig(task, start_trigger, ni.DAQmx_Val_Rising)

    # Set up the buffer size in advance
    if buffer_size is not None:
        log.debug('Setting buffer size to %d based on buffer_size', buffer_size)
        ni.DAQmxCfgOutputBuffer(task, buffer_size)
    elif total_samples is not None:
        m = 'Setting buffer size to %d based on total_samples'
        log.debug(m, total_samples)
        ni.DAQmxCfgOutputBuffer(task, total_samples)

    if total_samples is None:
        ni.DAQmxCfgSampClkTiming(task, sample_clock, fs, ni.DAQmx_Val_Rising,
                                 ni.DAQmx_Val_ContSamps, int(fs))
        log.debug('Configured continuous output')
    else:
        ni.DAQmxCfgSampClkTiming(task, sample_clock, fs, ni.DAQmx_Val_Rising,
                                 ni.DAQmx_Val_FiniteSamps, total_samples)
        log.debug('Configured finite output with %d samples', total_samples)

    # Store a reference to the callback pointers on the task object itself. This
    # prevents the callback pointers from getting garbage-collected. If they get
    # garbage-collected, then the callback no longer exists!
    if callback is not None:
        task._cb_ptr = create_everynsamples_callback(callback, callback_samples,
                                                     task, 'output')
    if done_callback is not None:
        task._done_cb_ptr = create_done_callback(done_callback, task)


    # Log configuration info regarding task
    result = ctypes.c_uint32()
    ni.DAQmxGetBufOutputBufSize(task, result)
    log.debug('AO buffer size %d', result.value)
    ni.DAQmxGetBufOutputOnbrdBufSize(task, result)
    log.debug('AO onboard buffer size %d', result.value)

    result = ctypes.c_double()
    ni.DAQmxGetAOGain(task, ao, result)
    log.debug('AO gain for %s is %.2f', ao, result.value)

    # Commit the tasks so we can catch resource errors early
    ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)

    return task


def create_done_callback(callback, task):
    def event_cb(task, status, data):
        callback()
        return 0
    log.debug('Configuring done callback')
    cb_ptr = ni.DAQmxDoneEventCallbackPtr(event_cb)
    ni.DAQmxRegisterDoneEvent(task, 0, cb_ptr, None)
    return cb_ptr


def create_everynsamples_callback(callback, samples, task, task_type='input'):
    def event_cb(task, event_type, n_samples, data):
        callback()
        return 0
    log.debug('Configuring every N samples callback with %d samples', samples)
    cb_ptr = ni.DAQmxEveryNSamplesEventCallbackPtr(event_cb)
    if task_type == 'input':
        event_type = ni.DAQmx_Val_Acquired_Into_Buffer
    elif task_type == 'output':
        event_type = ni.DAQmx_Val_Transferred_From_Buffer
    else:
        raise ValueError('Unrecognized task type')
    ni.DAQmxRegisterEveryNSamplesEvent(task, event_type, int(samples), 0,
                                       cb_ptr, None)
    return cb_ptr


def create_change_detect_callback(callback, rising=None, falling=None,
                                  task=None):
    '''
    Set up change detection on line(s) with Python callback

    Parameters
    ----------
    callback : callable
        Python function to call on event.
    rising : {None, str}
        Line to monitor for rising edge
    falling : {None, str}
        Line to monitor for falling edge
    '''
    if task is None:
        task = create_task()
    if rising is None and falling is None:
        raise ValueError('Must provide at least one line')
    if rising is None:
        rising = ''
    if falling is None:
        falling = ''

    log.debug('Creating change detect callback (rising: %s, falling: %s)',
              rising, falling)
    line = rising if rising == falling else ','.join((rising, falling))
    ni.DAQmxCreateDIChan(task, line, '', ni.DAQmx_Val_ChanForAllLines)
    ni.DAQmxCfgChangeDetectionTiming(task, rising, falling,
                                     ni.DAQmx_Val_ContSamps, 100)

    # Pointers to classmethods are not supported by ctypes, so we need to take
    # advantage of function closure to maintain a reference to self.  Must
    # return 0 to keep NIDAQmx happy (NIDAQmx expects all functions to return 0
    # to indicate success.  A non-zero return value indicates there was a
    # nerror).
    def event_cb(task, signal_ID, data):
        callback()
        return 0

    # Binding the pointer to an attribute on self seems to be necessary
    # to ensure the callback function does not disappear (presumably it
    # gets garbage-collected otherwise)
    _change_detect_callback_ptr = ni.DAQmxSignalEventCallbackPtr(event_cb)
    ni.DAQmxRegisterSignalEvent(task, ni.DAQmx_Val_ChangeDetectionEvent, 0,
                                _change_detect_callback_ptr, None)

    ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)
    return task, _change_detect_callback_ptr


def samples_available(task, epoch_size=None):
    result = ctypes.c_uint32()
    ni.DAQmxGetReadAvailSampPerChan(task, result)
    samples = result.value
    if epoch_size is not None:
        samples = int(np.floor(samples/epoch_size)*epoch_size)
    return samples


def num_channels(task):
    result = ctypes.c_uint32()
    ni.DAQmxGetTaskNumChans(task, result)
    return result.value


def channel_names(task):
    buftype = ctypes.c_char * 1024
    buf = buftype()
    ni.DAQmxGetTaskChannels(task, buf, 1024)
    return [c.strip() for c in buf.value.split(',')]


################################################################################
# Primary interface for DAQ
################################################################################
class DAQmxBase(object):

    # State can be uninitialized, initialized, running

    def __init__(self, *args, **kwargs):
        self._tasks = []
        self._state = 'uninitialized'
        super(DAQmxBase, self).__init__(*args, **kwargs)

    def start(self):
        log.debug('Starting %s', self.__class__.__name__)
        if self._state == 'uninitialized':
            self.setup()
        self._state = 'running'
        for task in self._tasks:
            ni.DAQmxStartTask(task)

    def stop(self):
        log.debug('Stopping %s', self)
        self._state = 'initialized'
        for task in self._tasks:
            try:
                ni.DAQmxStopTask(task)
            except ni.DAQError:
                pass
        if hasattr(self, 'complete_callback') \
                and self.complete_callback is not None:
            self.complete_callback()

    def clear(self):
        log.debug('Clearing %s', self.__class__.__name__)
        for task in self._tasks:
            try:
                ni.DAQmxClearTask(task)
            except ni.DAQError:
                pass
        self._state = 'uninitialized'

    def setup(self):
        log.debug('Setting up %s', self.__class__.__name__)
        self._state = 'initialized'

    def __del__(self):
        if self._state != 'uninitialized':
            self.clear()

    def __enter__(self):
        self.setup()
        self.start()
        return self

    def __exit__(self):
        self.__del__()


class DAQmxInput(DAQmxBase):

    DIFF = ni.DAQmx_Val_Diff
    NRSE = ni.DAQmx_Val_NRSE
    RSE = ni.DAQmx_Val_RSE

    def __init__(self, fs=25e3, input_line='/PXI4461/ai0', callback=None,
                 callback_samples=None, expected_range=10, run_line=None,
                 pipeline=None, complete_callback=None, start_trigger=None,
                 record_mode=ni.DAQmx_Val_Diff):
        for k, v in locals().items():
            setattr(self, k, v)
        super(DAQmxInput, self).__init__()

    def setup(self):
        log.debug('Setting up continuous AI tasks')
        self._task_analog = create_ai(ai=self.input_line,
                                      fs=self.fs,
                                      expected_range=self.expected_range,
                                      callback=self.trigger_callback,
                                      callback_samples=self.callback_samples,
                                      start_trigger=self.start_trigger,
                                      record_mode=self.record_mode)

        self._tasks.append(self._task_analog)
        self.channels = num_channels(self._task_analog)
        log.debug('Configured continous AI with %d channels controlled by %s',
                  self.channels, self.run_line)
        super(DAQmxInput, self).setup()
        self._event = threading.Event()

    def samples_available(self):
        if self.callback_samples is not None:
            return self.callback_samples
        else:
            return samples_available(self._task_analog)

    def read_analog(self, timeout=None):
        if timeout is None:
            timeout = 0
        samps_per_chan = self.samples_available()
        result = ctypes.c_long()
        analog_data = np.empty((self.channels, samps_per_chan), dtype=np.double)
        ni.DAQmxReadAnalogF64(self._task_analog, samps_per_chan, timeout,
                              ni.DAQmx_Val_GroupByChannel, analog_data,
                              analog_data.size, result, None)
        log.debug('Read %d s/chan from %s', result.value, self.input_line)
        return analog_data.reshape((1, self.channels, -1))

    def trigger_callback(self):
        log.debug('Trigger callback')
        waveforms = self.read_analog()
        try:
            if self.callback is not None:
                self.callback(waveforms)
                log.debug('Sent to provided callback')
            if self.pipeline is not None:
                self.pipeline.send(waveforms)
                log.debug('Sent to provided pipeline')
        except GeneratorExit:
            log.debug('Captured generator exit event')
            self._event.set()
            self.stop()
            self.complete = True

    def join(self, timeout=None):
        return self._event.wait(timeout)

    def start(self):
        super(DAQmxInput, self).start()
        self._event.clear()

    def stop(self):
        super(DAQmxInput, self).stop()
        self._event.set()


class DAQmxChannel(Channel):

    # Based on testing of the PGA2310/OPA234/BUF634 circuit with volume control
    # set to 0 dB, this is the maximum allowable voltage without clipping.
    voltage_min = -1.1*np.sqrt(2)
    voltage_max = 1.1*np.sqrt(2)

    attenuator = ScalarInput(default=None, unit=None)
    attenuator_channel = ScalarInput(default=0, unit=None)

    def set_gain(self, gain):
        return self.set_attenuation(-gain)

    def set_attenuation(self, attenuation):
        self.attenuator.set_attenuation(attenuation, self.attenuator_channel)
        self.attenuation = attenuation

    def get_nearest_attenuation(self, atten):
        if self.attenuator is None:
            raise ValueError('Channel does not have attenuation control')
        return self.attenuator.get_nearest_attenuation(atten)


def residual_sf(requested, actual):
    result = [10**((a-r)/20.0) for r, a in zip(requested, actual)]
    return np.array(result)


class DAQmxOutput(DAQmxBase):

    def __init__(self, fs=200e3,
                 output_line=DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
                 expected_range=DAQmxDefaults.AO_RANGE, buffer_size=2,
                 monitor_interval=0.5, duration=None, done_callback=None,
                 start_trigger=None, attenuations=None, **kwargs):

        variables = locals()
        kwargs = variables.pop('kwargs')
        for k, v in variables.items():
            setattr(self, k, v)
        super(DAQmxOutput, self).__init__(**kwargs)
        self.setup()

    def setup(self):
        log.debug('Setting up AO tasks')

        kwargs = {}
        if hasattr(self, 'monitor') and self.monitor_interval is not None:
            self.buffer_samples = int(self.fs*self.buffer_size)
            self.monitor_samples = int(self.fs*self.monitor_interval)
            log.debug('Buffer size %d, monitor every %d samples',
                    self.buffer_samples, self.monitor_samples)
            kwargs['callback'] = self.monitor
            kwargs['callback_samples'] = self.monitor_samples
            kwargs['total_samples'] = None
            kwargs['buffer_size'] = self.buffer_samples
        else:
            kwargs['total_samples'] = int(self.duration*self.fs)

        self._task_analog = create_ao(self.output_line,
                                      self.fs,
                                      self.expected_range,
                                      done_callback=self._done_callback,
                                      start_trigger=self.start_trigger,
                                      **kwargs)

        self._tasks.append(self._task_analog)
        self.samples_written = 0
        self.n_channels = num_channels(self._task_analog)
        self.channel_names = channel_names(self._task_analog)

        log.debug('Configured AO with %d channels', self.n_channels)

        # Set up the attenuation. If we cannot achieve the desired attenuation,
        # the waveform will have to be scaled appropriately before writing.
        self.attenuator = DAQmxBaseAttenuator()
        if self.attenuations is not None:
            attenuations = \
                self.attenuator.get_nearest_attenuations(self.attenuations)
            self.attenuator.set_attenuations(attenuations)
            self.waveform_sf = residual_sf(self.attenuations, attenuations)
        else:
            self.waveform_sf = np.array([1]*self.n_channels)
        super(DAQmxOutput, self).setup()

    def get_write_size(self):
        if self._state == 'halted':
            return 0
        if self.samples_written == 0:
            log.debug('initial write size %d', self.buffer_samples)
            return self.buffer_samples
        result = ctypes.c_uint64()
        ni.DAQmxGetWriteTotalSampPerChanGenerated(self._task_analog, result)
        buffered = self.samples_written-result.value
        write_size = self.buffer_samples-buffered
        log.debug('%d samples written to buffer, '
                  '%d samples transferred to device '
                  '%d samples to add to buffer',
                  self.samples_written, result.value, write_size)
        return write_size

    def write(self, analog, iti=0):
        analog = np.asarray(analog)*self.waveform_sf[..., np.newaxis]
        if iti:
            samples = int(iti*self.fs)
            analog = np.pad(analog, ((0, 0), (0, samples)), 'constant',
                            constant_values=0)
        analog_result = ctypes.c_long()
        log.debug('Writing array of shape %r to AO', analog.shape)
        log.debug('First sample %f, last sample %f', analog.ravel()[0],
                  analog.ravel()[-1])
        ni.DAQmxWriteAnalogF64(self._task_analog, analog.shape[-1], False, -1,
                               ni.DAQmx_Val_GroupByChannel,
                               analog.ravel().astype(np.double), analog_result,
                               None)
        log.debug('Wrote %d s/chan to %s', analog_result.value,
                  self.output_line)
        self.samples_written += analog_result.value

    def status(self):
        try:
            result = ctypes.c_ulong()
            ni.DAQmxIsTaskDone(self._task_analog, result)
        except:
            return self.samples_written
        result = ctypes.c_uint64()
        ni.DAQmxGetWriteTotalSampPerChanGenerated(self._task_analog, result)
        log.debug('%d s/chan generated', result.value)
        return result.value

    def complete(self):
        if self.samples_written != 0:
            return self.status() == self.samples_written
        else:
            return False

    def get_nearest_attenuation(self, attenuation):
        if self.attenuator is None:
            raise ValueError('Cannot control attenuation')
        return self.attenuator.get_nearest_attenuation(attenuation)

    def _done_callback(self):
        if self.done_callback is not None:
            self.done_callback()


class QueuedDAQmxPlayer(DAQmxOutput, QueuedPlayer):

    pass


class ContinuousDAQmxPlayer(DAQmxOutput, ContinuousPlayer):

    def clear(self):
        super(ContinuousDAQmxPlayer, self).clear()
        self.reset()


################################################################################
# Attenuation support
################################################################################
class DAQmxBaseAttenuator(DAQmxBase):

    ATTENUATION_STEPS = [0]
    ATTENUATION_CHANNELS = len(DAQmxDefaults.SPEAKER_OUTPUTS)

    def set_gain(self, gain, channel=None):
        self.set_attenuation(-gain, channel)

    def set_attenuation(self, attenuation, channel=None):
        if channel is None:
            attenuations = [attenuation]*self.ATTENUATION_CHANNELS
        else:
            attenuations = [None]*self.ATTENUATION_CHANNELS
            attenuations[channel] = attenuation
        self.set_attenuations(attenuations)

    def set_gains(self, gains):
        self.set_attenuations([-g for g in gains])

    def set_attenuations(self, attenuations):
        self._validate_attenuations(attenuations)

    def get_nearest_attenuation(self, attenuation):
        attenuation_steps = np.array(self.ATTENUATION_STEPS)
        delta = attenuation-attenuation_steps
        nearest_attenuation = attenuation_steps[delta >= 0].max()
        #residual_scaling_factor = 10**((nearest_attenuation-attenuation)/20.0)
        #return nearest_attenuation, residual_scaling_factor
        return nearest_attenuation

    def get_nearest_attenuations(self, attenuations):
        #results = [self.get_nearest_attenuation(a) for a in attenuations]
        #nearest_attenuations, residual_scaling_factors = zip(*results)
        #return nearest_attenuations, residual_scaling_factors
        return [self.get_nearest_attenuation(a) for a in attenuations]

    def _validate_attenuations(self, attenuations):
        for a in attenuations:
            if (a is not None) and (a not in self.ATTENUATION_STEPS):
                raise ValueError('Unsupported attenuation {}'.format(a))

    def set_mute(self, mute):
        if mute:
            raise ValueError('Cannot mute output')


class DAQmxPGA2310Attenuator(DAQmxBaseAttenuator):
    '''
    Handles configuration and control of niDAQmx for controlling volume via
    serial control of a programmable IC chip
    '''
    # Gain control settings in dB.  This is IC specific.  Maximum gain is
    # 31.5 dB and max atten is 96 dB.
    VOLUME_STEP = 0.5
    VOLUME_MAX = 31.5
    VOLUME_MIN = -50
    VOLUME_BITS = 16

    MAX_ATTEN = -VOLUME_MIN
    MIN_ATTEN = -VOLUME_MAX
    ATTEN_STEP = VOLUME_STEP

    ATTEN_STEPS = np.arange(MIN_ATTEN, MAX_ATTEN+ATTEN_STEP, ATTEN_STEP)

    fs = 200e3

    def __init__(self, clk_line, cs_line, sdi_line, mute_line, zc_line):
        '''
        Parameters
        ----------
        clk_line : str
            Line to use for serial clock (rising edge of clock tells IC to read
            the state of the data line)
        cs_line : str
            Line to use for chip select (low value indicates that data is being
            written).
        sdi_line : str
            Line to use for serial data.
        mute_line : str
            Line to use for controlling mute state.  IC is muted when line is
            high.
        zc_line : str
            Line to use for controlling zero crossing state.  When high, IC
            will change attenuation only when a zero crossing of the signal
            occurs.
        '''
        for k, v in locals().items():
            setattr(self, k, v)
        self._right_setting = None
        self._left_setting = None
        raise ValueError
        super(DAQmxPGA2310Atten, self).__init__()

    def setup(self, gain=-np.inf):
        # Configure the tasks and IO lines
        log.debug('Configuring NI tasks')
        self._task_mute = create_task()
        self._task_zc = create_task()
        self._task_clk = create_task()
        self._task_cs = create_task()
        self._task_sdi = create_task()

        # Set up digital output lines.  In theory we can combine both of these
        # lines into a single task, but it's easier to program them separately.
        ni.DAQmxCreateDOChan(self._task_mute, self.mute_line, 'mute',
                             ni.DAQmx_Val_ChanPerLine)
        ni.DAQmxCreateDOChan(self._task_zc, self.zc_line, 'zc',
                             ni.DAQmx_Val_ChanPerLine)
        ni.DAQmxCreateDOChan(self._task_clk, self.clk_line, 'clk',
                             ni.DAQmx_Val_ChanPerLine)
        ni.DAQmxCreateDOChan(self._task_cs, self.cs_line, 'cs',
                             ni.DAQmx_Val_ChanPerLine)
        ni.DAQmxCreateDOChan(self._task_sdi, self.sdi_line, 'sdi',
                             ni.DAQmx_Val_ChanPerLine)

        # Use the soft mute option
        self.set_mute(False)
        self.set_zero_crossing(False)
        if np.isscalar(gain):
            self.set_gains(gain)
        else:
            self.set_gain(*gain)
        super(DAQmxPGA2310Atten, self).setup()

    def start(self):
        super(DAQmxPGA2310Atten, self).start()
        self._send_bit(self._task_cs, 1)
        self.set_mute(False)
        self.set_zero_crossing(False)
        self.set_atten(np.inf, np.inf)

    def clear(self):
        for task in (self._task_zc, self._task_mute, self._task_clk,
                     self._task_cs, self._task_sdi):
            ni.DAQmxClearTask(task)
        super(DAQmxPGA2310Atten, self).clear()

    def _gain_to_byte(self, gain):
        '''
        Compute the byte value for the requested gain

        This is chip-specific.  The TI PGA2311 volume control chip uses the
        formula gain (dB) = 3.15-(0.5*(255-byte)).

        Parameters
        ----------
        gain : float
            Desired gain in dB.  For attenuation, pass a negative value.  To
            mute the channel, pass -np.inf.

        Returns
        -------
        byte : int
            Byte value to send via serial interface.

        Raises
        ------
        ValueError if the requested gain is outside valid boundaries or is not
        a multiple of the gain step size.
        '''
        if gain == -np.inf:
            return 0
        if gain > self.VOLUME_MAX:
            raise ValueError('Requested gain is too high')
        if gain < self.VOLUME_MIN:
            raise ValueError('Requested gain is too low')
        if gain % self.VOLUME_STEP != 0:
            raise ValueError('Requested gain is not multiple of step size')
        return int(255-((31.5-gain)/0.5))

    def _byte_to_gain(self, byte):
        return 31.5-(0.5*(255-byte))

    def _gain_to_bits(self, right, left):
        rbyte = self._gain_to_byte(right)
        lbyte = self._gain_to_byte(left)
        word = (rbyte << 8) | lbyte
        return get_bits(word, self.VOLUME_BITS, 'big-endian')

    def _send_bit(self, task, bit):
        ni.DAQmxWriteDigitalLines(task, 1, True, 0,
                                  ni.DAQmx_Val_GroupByChannel,
                                  np.array([bit], dtype=np.uint8),
                                  ctypes.c_int32(), None)

    def _send_bits(self, bits):
        self._send_bit(self._task_cs, 0)
        log.debug('Sending %r to com line', bits)
        for bit in bits:
            self._send_bit(self._task_sdi, bit)
            self._send_bit(self._task_clk, 1)
            self._send_bit(self._task_clk, 0)
        self._send_bit(self._task_cs, 1)

    def set_gain(self, right=None, left=None):
        '''
        Set the IC volume control to the desired gain.

        This function will not return until the serial value has been written
        to the digital output.  There is no way to programatically check that
        the IC is configured properly.

        If -np.inf is provided, the channel will be muted.  If None is provided,
        the setting for the channel will not change.

        Parameters
        ----------
        right : {None, float, -np.inf}
            Desired gain for right channel in dB.  For attenuation, pass a
            negative value.
        left : {None, float, -np.inf}i
            Desired gain for left channel in dB.  For attenuation, pass a
            negative value.

        Returns
        -------
        None

        Raises
        ------
        ValueError if the requested gain is outside valid boundaries or is not
        a multiple of the gain step size.
        '''
        log.debug('Setting gain to %r (right) and %r (left)', right, left)
        if right is None:
            right = self._right_setting
        if left is None:
            left = self._left_setting
        bits = self._gain_to_bits(right, left)
        self._send_bits(bits)
        self._right_setting = right
        self._left_setting = left

    def set_gains(self, gain):
        '''
        Utility function for setting gain of right and left to same value
        '''
        self.set_gain(gain, gain)

    def set_atten(self, right=None, left=None):
        '''
        Set the IC volume control to the desired attenuation.

        This is a convenience method that allows one to specify the volume as an
        attenuation rather than a gain as it passes the inverted values to
        `set_gain`.

        .. note::
            Be sure to set attenuation to np.inf if you want to mute the
            channel instead of -np.inf.
        '''
        if right is not None:
            right = -right
        if left is not None:
            left = -left
        self.set_gain(right, left)

    def set_mute(self, mute):
        '''
        Set the IC volume mute to the desired value.

        Parameters
        ----------
        mute : bool
            Mute the volume?

        This function will not return until the serial value has been written to
        the digital output.  There is no way to programatically check that the
        IC is configured properly.
        '''
        log.debug('Setting volume mute to %s', mute)
        # A LOW setting on the DIO line indicates that the chip should mute the
        # output so we need to invert the value of the mute flag.
        ni.DAQmxWriteDigitalLines(self._task_mute, 1, True, -1,
                                  ni.DAQmx_Val_GroupByChannel,
                                  np.array([not mute], dtype=np.uint8),
                                  ctypes.c_int32(), None)
        ni.DAQmxWaitUntilTaskDone(self._task_mute, 1)

    def set_zero_crossing(self, zero_crossing):
        '''
        Set the IC zero crossing control the desired value.

        When True, the gain will change only when the signal crosses zero to
        prevent unwanted clicks or transients.

        Parameters
        ----------
        zero_crossing : bool
            Change gain only on zero crossing?

        This function will not return until the serial value has been written to
        the digital output.  There is no way to programatically check that the
        IC is configured properly.
        '''
        log.debug('Setting zero crossing to %s', zero_crossing)
        ni.DAQmxWriteDigitalLines(self._task_zc, 1, True, -1,
                                  ni.DAQmx_Val_GroupByChannel,
                                  np.array([zero_crossing], dtype=np.uint8),
                                  ctypes.c_int32(), None)
        ni.DAQmxWaitUntilTaskDone(self._task_zc, 1)


def class_for_name(full_name):
    module_name, class_name = full_name.rsplit('.', 1)
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def get_default_attenuator(**kwargs):
    c = class_for_name(DAQmxDefaults.ATTEN_CLASS)
    class_kw = getattr(DAQmxDefaults, 'ATTEN_KW', {})
    class_kw.update(kwargs)
    return c(**class_kw)


################################################################################
# Primary helpers for DAQ
################################################################################
class BaseDAQmxAcquire(object):

    def stop(self):
        self.iface_adc.clear()
        self.iface_dac.clear()

    def join(self):
        while not self.complete:
            time.sleep(0.1)

    def poll(self, waveforms):
        log.debug('Received data')
        lb, ub = self.epochs_acquired, self.epochs_acquired+len(waveforms)
        self.waveforms.append(waveforms)
        self.epochs_acquired += len(waveforms)
        if self.epochs_acquired >= self.repetitions:
            log.debug('Halting acquisition')
            self.stop()
            self.complete = True
        if self.callback is not None:
            self.callback(self.epochs_acquired, self.complete)
        log.debug('Completed processing received data')

    def get_waveforms(self, remove_iti=False):
        waveforms = np.concatenate(self.waveforms, axis=0)
        if remove_iti:
            trim_samples = int(self.iti*self.adc_fs)
            waveforms = waveforms[..., :-trim_samples]
        return waveforms


class DAQmxAcquire(BaseDAQmxAcquire):

    def __init__(self, channels, repetitions, output_line, input_line, gain,
                 dac_fs=100e3, adc_fs=100e3, duration=1, iti=0.01,
                 input_range=10, callback=None, waveform_buffer=None):

        log.debug('Setting up acquisition')
        for k, v in locals().items():
            setattr(self, k, v)

        self.complete = False
        self.waveforms = []
        self.epochs_acquired = 0

        # Set up the recording
        self.iface_adc = DAQmxInput(fs=adc_fs,
                                    callback_samples=int(duration*adc_fs),
                                    input_line=input_line,
                                    expected_range=input_range,
                                    start_trigger='ao/StartTrigger',
                                    callback=self.poll)
        self.iface_adc.start()

        self.iface_atten = get_default_attenuator()
        self.iface_atten.setup()
        if np.isscalar(gain):
            self.iface_atten.set_gains(gain)
        else:
            self.iface_atten.set_gain(*gain)
        self.iface_atten.clear()

        if waveform_buffer is None:
            size = (repetitions, self.iface_adc.channels, int(duration*adc_fs))
            waveform_buffer = np.empty(size, dtype=np.float32)
        self.waveform_buffer = waveform_buffer

        self.iface_dac = QueuedDAQmxPlayer(fs=dac_fs, duration=duration,
                                           output_line=output_line)
        for channel in channels:
            self.iface_dac.add_channel(channel)
        self.iface_dac.queue_init('FIFO')
        self.iface_dac.queue_append(repetitions, iti)

    def start(self):
        log.debug('Starting acquisition')
        self.iface_dac.play_queue()


def acquire(*args, **kwargs):
    daq = DAQmxAcquire(*args, **kwargs)
    daq.start()
    daq.join()
    return daq.get_waveforms()


class DAQmxAcquireWaveform(BaseDAQmxAcquire):

    def __init__(self, waveform, repetitions, output_line, input_line, gain,
                 dac_fs=100e3, adc_fs=100e3, iti=0.01, input_range=10,
                 output_range=10, callback=None):

        log.debug('Setting up acquisition')
        for k, v in locals().items():
            setattr(self, k, v)

        self.waveform = waveform
        self.iti = iti
        self.complete = False
        self.waveforms = []
        self.epochs_acquired = 0

        waveform_duration = waveform.shape[-1]/dac_fs
        trial_duration = waveform_duration+iti
        total_duration = trial_duration*repetitions
        epoch_duration = trial_duration-adc_fs**-1
        callback_samples = int(trial_duration*adc_fs)

        # Set up the recording
        self.iface_adc = DAQmxInput(fs=adc_fs,
                                    callback_samples=callback_samples,
                                    input_line=input_line,
                                    expected_range=input_range,
                                    start_trigger='ao/StartTrigger',
                                    callback=self.poll)

        self.iface_dac = DAQmxOutput(fs=dac_fs,
                                     duration=total_duration,
                                     output_line=output_line,
                                     expected_range=output_range,
                                     attenuations=[-gain])

        # Write all data to the buffer before starting since this is known in
        # advance. This does not work for an infinite number of repetitions
        # (obviously).
        for i in range(repetitions):
            self.iface_dac.write(waveform, iti)

    def start(self):
        log.debug('Starting acquisition')
        self.iface_adc.start()
        self.iface_dac.start()


def acquire_waveform(*args, **kwargs):
    daq = DAQmxAcquireWaveform(*args, **kwargs)
    daq.start()
    daq.join()
    return daq.get_waveforms()


################################################################################
# Debugging functions
################################################################################
def reset(device=DAQmxDefaults.DEV):
    ni.DAQmxResetDevice(device)
