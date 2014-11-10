'''
Classes for configuring and recording using NIDAQmx compatible devices
'''

# TODO setup DAQmxRegisterDoneEvent callback

from __future__ import division

import ctypes
import unittest
import time

import PyDAQmx as ni
import numpy as np

from neurogen.channel import Channel
from neurogen.player import Player, QueuedPlayerMixin, ThreadPlayerMixin
from neurogen.blocks import ScalarInput

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


################################################################################
# Current configuration of niDAQmx hardware
################################################################################
class DAQmxDefaults(object):
    '''
    Define defaults for configuring the niDAQmx tasks.  Often we have lines
    hard-wired for certain purposes, so we need to use these lines.
    '''
    DEV = 'Dev1'

    # See documentation in nidaqmx.TriggeredDAQmxSource for information on the
    # specifics of each line.
    ERP_INPUT = '/{}/ai0'.format(DEV)
    MIC_INPUT = '/{}/ai1'.format(DEV)
    REF_MIC_INPUT = '/{}/ai2'.format(DEV)

    AI_COUNTER = '/{}/Ctr0'.format(DEV)
    AI_TRIGGER = '/{}/PFI1'.format(DEV)
    AI_RUN = None
    AI_RANGE = 10

    AO_TRIGGER = '/{}/port0/line0'.format(DEV)
    AO_RUN = '/{}/port0/line2'.format(DEV)
    AO_RANGE = 10
    DUAL_SPEAKER_OUTPUT = '/{}/ao0:1'.format(DEV)
    LEFT_SPEAKER_OUTPUT = '/{}/ao0'.format(DEV)
    RIGHT_SPEAKER_OUTPUT = '/{}/ao1'.format(DEV)

    VOLUME_CLK = '/{}/port0/line4'.format(DEV)
    VOLUME_CS = '/{}/port0/line1'.format(DEV)
    VOLUME_SDI = '/{}/port0/line6'.format(DEV)
    VOLUME_MUTE = '/{}/port1/line6'.format(DEV)
    VOLUME_ZC = '/{}/port1/line5'.format(DEV)
    DIO_CLOCK = '/{}/FreqOut'.format(DEV)

    TRIGGER_DURATION = 10e-3


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


def create_continuous_ai(ai, fs, expected_range=10, callback=None,
                         callback_samples=None, task_analog=None):
    if task_analog is None:
        task_analog = create_task()
    vlb, vub = -expected_range, expected_range
    ni.DAQmxCreateAIVoltageChan(task_analog, ai, '', ni.DAQmx_Val_RSE, vlb, vub,
                                ni.DAQmx_Val_Volts, '')
    ni.DAQmxCfgSampClkTiming(task_analog, None, fs, ni.DAQmx_Val_Rising,
                             ni.DAQmx_Val_ContSamps, 0)
    ni.DAQmxTaskControl(task_analog, ni.DAQmx_Val_Task_Commit)
    if callback is not None:
        self.create_everynsamples_callback(task_analog, callback,
                                            callback_samples)
    return task_analog


def create_retriggerable_ai(ai, fs, epoch_size, trigger, expected_range=10,
                            counter=None, run=None, callback=None,
                            task_analog=None, task_record=None,
                            trigger_duration=DAQmxDefaults.TRIGGER_DURATION):
    '''
    Create retriggerable data acquisition

    Parameters
    ----------
    ai : str
        Line to acquire data from
    trigger : str
        Line to monitor for trigger
    counter : str
        Counter to use for controlling data acquisition.  Note that this
        type of data acquisition requires the use of paired counters.  Ctr0
        will be paired with Ctr1, Ctr2 will be paired with Ctr3, etc.  If
        your device only has two counters, then you cannot use an event
        timer as well.  The counters are hardwired together (i.e. if you
        specify '/Dev1/Ctr0' or '/Dev1/Ctr1', the other counter in the pair
        will automatically be reserved).
    task_analog : {None, niDAQmx task}
        Task to use for reading analog data in.  If None, a new task will be
        created.
    task_record : {None, niDAQmx task}
        Task to use for controlling record counter.  If None, a new task
        will be created.

    Returns
    -------
    task_analog
        Configured and committed task for analog data
    task_record
        Configured and committed task for triggering acquisition of analog
        data.
    cb_ptr
        Pointer to the niDAQmx C callback.  This needs to be kept in program
        memory by keeping it assigned to a variable name otherwise the callback
        will get garbage-collected.

    Limitations
    -----------
    Triggered data acquisition on M-series cards requires two counters (the
    second counter is implied based on which one is hard-wired to the specified
    counter, see device manual for details).  This means that if your device
    has only two counters, you cannot use the second counter to generate
    hardware-timed timestamps.

    Task implementation details
    ---------------------------
    This uses an onboard device counter which is triggered by the rising edge
    of a digital trigger (must be a PFI or ChangeDetectionEvent).  The counter
    will generate a series of pulses for the specified acquisition duration at
    the specified acquisition rate.  This allows us to continuously spool
    epoch-based data to a buffer and supports very high rates of epoch
    acquisition without risk of losing data.

    Newer cards (e.g. X-series) support retriggerable tasks, so the approach
    used here may not be necessary; however, it depends on how quickly the
    X-series tasks can rearm the trigger.
    '''
    if task_analog is None:
        task_analog = create_task()
    if task_record is None:
        task_record = create_task()
    vlb, vub = -expected_range, expected_range

    # Set up the analog input to continously acquire samples.  However,
    # since it's tied to the internal output of the counter, it will only
    # acquire when the counter is active.  Since the counter only runs for a
    # finite number of samples after each trigger, this is effectively a
    # triggered
    ni.DAQmxCreateAIVoltageChan(task_analog, ai, '', ni.DAQmx_Val_RSE, vlb, vub,
                                ni.DAQmx_Val_Volts, '')
    ni.DAQmxCfgSampClkTiming(task_analog, counter+'InternalOutput', fs,
                             ni.DAQmx_Val_Rising, ni.DAQmx_Val_ContSamps,
                             epoch_size)
    ni.DAQmxSetBufInputBufSize(task_analog, epoch_size*1000)

    ni.DAQmxCreateCOPulseChanFreq(task_record, counter, '', ni.DAQmx_Val_Hz,
                                  ni.DAQmx_Val_Low, 0, fs, 0.5)
    ni.DAQmxCfgImplicitTiming(task_record, ni.DAQmx_Val_FiniteSamps, epoch_size)
    ni.DAQmxCfgDigEdgeStartTrig(task_record, trigger, ni.DAQmx_Val_Rising)

    # Ensure trigger rearms after each detection event
    ni.DAQmxSetStartTrigRetriggerable(task_record, True)
    ni.DAQmxSetDigEdgeStartTrigDigFltrMinPulseWidth(task_record,
                                                    trigger_duration*0.5)

    ni.DAQmxTaskControl(task_analog, ni.DAQmx_Val_Task_Commit)
    ni.DAQmxTaskControl(task_record, ni.DAQmx_Val_Task_Commit)

    if callback is not None:
        cb_ptr = create_everynsamples_callback(callback, epoch_size,
                                               task_analog)
    else:
        cb_ptr = None

    return task_analog, task_record, cb_ptr


def create_continuous_ao(ao, trigger, run, fs,
                         expected_range=DAQmxDefaults.AO_RANGE,
                         task_analog=None, task_digital=None):
    if task_analog is None:
        task_analog = create_task()
    if task_digital is None:
        task_digital = create_task()
    vmin, vmax = -expected_range, expected_range

    # Setup analog output
    ni.DAQmxCreateAOVoltageChan(task_analog, ao, '', vmin, vmax,
                                ni.DAQmx_Val_Volts, '')
    ni.DAQmxSetWriteRegenMode(task_analog, ni.DAQmx_Val_DoNotAllowRegen)
    ni.DAQmxCfgSampClkTiming(task_analog, '', fs, ni.DAQmx_Val_Rising,
                             ni.DAQmx_Val_ContSamps, int(fs))

    # Set up trigger line
    do = ','.join([trigger, run])
    ni.DAQmxCreateDOChan(task_digital, do, '', ni.DAQmx_Val_ChanPerLine)
    ni.DAQmxSetWriteRegenMode(task_digital, ni.DAQmx_Val_DoNotAllowRegen)
    ni.DAQmxCfgSampClkTiming(task_digital, 'ao/SampleClock', fs,
                             ni.DAQmx_Val_Rising, ni.DAQmx_Val_ContSamps,
                             int(fs))

    # Commit the tasks so we can catch resource errors early
    ni.DAQmxTaskControl(task_analog, ni.DAQmx_Val_Task_Commit)
    ni.DAQmxTaskControl(task_digital, ni.DAQmx_Val_Task_Commit)

    # Log configuration info regarding task
    result = ctypes.c_uint32()
    ni.DAQmxGetBufOutputBufSize(task_analog, result)
    log.debug('AO buffer size %d', result.value)
    ni.DAQmxGetBufOutputOnbrdBufSize(task_analog, result)
    log.debug('AO onboard buffer size %d', result.value)
    ni.DAQmxGetBufOutputBufSize(task_digital, result)
    log.debug('DO buffer size %d', result.value)
    ni.DAQmxGetBufOutputOnbrdBufSize(task_digital, result)
    log.debug('DO onboard buffer size %d', result.value)

    return task_analog, task_digital


def create_everynsamples_callback(callback, samples, task):
    def event_cb(task, event_type, n_samples, data):
        callback()
        return 0
    log.debug('Configuring every N samples callback with %d samples', samples)
    cb_ptr = ni.DAQmxEveryNSamplesEventCallbackPtr(event_cb)
    event_type = ni.DAQmx_Val_Acquired_Into_Buffer
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


################################################################################
# Primary interface for DAQ
################################################################################
class DAQmxBase(object):

    def __init__(self):
        self._tasks = []
        self._state = 'initialized'

    def start(self):
        self._state = 'running'
        for task in self._tasks:
            ni.DAQmxStartTask(task)

    def stop(self):
        self._state = 'halted'
        for task in self._tasks:
            try:
                ni.DAQmxStopTask(task)
            except ni.DAQError:
                pass

    def clear(self):
        self._state = 'halted'
        for task in self._tasks:
            try:
                ni.DAQmxClearTask(task)
            except ni.DAQError:
                pass


class DAQmxSource(DAQmxBase):
    '''
    Read data from device
    '''
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
        return analog_data.reshape((self.channels, -1))


class ContinuousDAQmxSource(DAQmxSource):

    def __init__(self, fs=25e3, input_line='/Dev1/ai0', callback=None,
                 callback_samples=None, expected_range=None):
        '''
        Parameters
        ----------
        setup : bool
            Automatically run setup procedure.  If False, you need to call
            `setup` method before you can acquire data.
        fs : float (Hz)
            Sampling frequency
        input_line : str
            Line to acquire data from
        callback : {None, callable}
            TODO
        '''
        super(ContinuousDAQmxSource, self).__init__()
        for k, v in locals().items():
            setattr(self, k, v)

    def setup(self):
        self._task_analog = create_continuous_ai(self.input_line, self.callback,
                                                 self.callback_samples)
        self._tasks.append(self._task_analog)
        self.channels = num_channels(self._task_analog)
        log.debug('Configured retriggerable AI with %d channels', self.channels)


class TriggeredDAQmxSource(DAQmxSource):

    def __init__(self, input_line, fs, epoch_duration,
                 trigger_line=DAQmxDefaults.AI_TRIGGER,
                 expected_range=DAQmxDefaults.AI_RANGE,
                 counter_line=DAQmxDefaults.AI_COUNTER,
                 run_line=DAQmxDefaults.AI_RUN,
                 trigger_duration=DAQmxDefaults.TRIGGER_DURATION,
                 callback=None):
        '''
        Parameters
        ----------
        setup : bool
            Automatically run setup procedure.  If False, you need to call
            `setup` method before you can acquire data.
        fs : float (Hz)
            Sampling frequency
        epoch_duration : float (sec)
            Duration of epoch to acquire
        input_line : str
            Line to acquire data from
        counter_line : str
            Line to reserve for retriggerable acquisition
        trigger_line : str
            Line to monitor for trigger
        run_line : {None, str}
            Line which indicates stimulus is in progress.  The task will end
            when this goes low.
        trigger_duration : float
            Duration of trigger.  Used to set debounce filter to prevent
            spurious triggers.
        callback : {None, callable}
            TODO
        '''
        super(TriggeredDAQmxSource, self).__init__()
        for k, v in locals().items():
            setattr(self, k, v)
        self.epoch_size = int(epoch_duration * fs)

    def setup(self):
        self._task_analog, self._task_digital, self._cb_ptr = \
            create_retriggerable_ai(self.input_line, self.fs, self.epoch_size,
                                    self.trigger_line, self.expected_range,
                                    self.counter_line, self.run_line,
                                    self.callback)
        self._tasks.extend((self._task_analog, self._task_digital))
        self.channels = num_channels(self._task_analog)
        log.debug('Configured retriggerable AI with %d channels', self.channels)

    def read_timer(self):
        result = ctypes.c_uint32()
        ni.DAQmxReadCounterScalarU32(self._task_counter, 0, result, None)
        return result.value

    def samples_available(self):
        return samples_available(self._task_analog, self.epoch_size)

    def read_analog(self, timeout=None):
        '''
        Returns 3D array, epoch, channel, time
        '''
        analog_data = super(TriggeredDAQmxSource, self).read_analog(timeout)
        epochs = int(analog_data.shape[-1]/self.epoch_size)
        analog_data.shape = self.channels, epochs, self.epoch_size
        return analog_data.swapaxes(0, 1)


class DAQmxChannel(Channel):

    # Based on testing of the PGA2310/OPA234/BUF634 circuit with volume control
    # set to 0 dB, this is the maximum allowable voltage without clipping.
    voltage_min = -1.5
    voltage_max = 1.5

    attenuator = ScalarInput(default=None, unit=None)
    attenuator_channel = ScalarInput(default=0, unit=None)

    def set_attenuation(self, atten):
        self.attenuator.set_atten(**{self.attenuator_channel: atten})
        self.attenuation = atten

    def get_nearest_attenuation(self, atten):
        if self.attenuator is None:
            raise ValueError('Channel does not have attenuation control')
        print 'getting nearest'
        return self.attenuator.get_nearest_atten(atten)


class DAQmxPlayer(DAQmxBase, ThreadPlayerMixin, QueuedPlayerMixin, Player):

    def __init__(self, fs=200e3, output_line='/Dev1/ao0',
                 expected_range=DAQmxDefaults.AO_RANGE,
                 trigger_line=DAQmxDefaults.AO_TRIGGER,
                 run_line=DAQmxDefaults.AO_RUN, attenuator=None, **kwargs):
        variables = locals()
        kwargs = variables.pop('kwargs')
        for k, v in variables.items():
            setattr(self, k, v)
        DAQmxBase.__init__(self)
        Player.__init__(self, **kwargs)
        self.write_size = int(self.fs * 1)

    def setup(self):
        self._task_analog, self._task_digital = \
            create_continuous_ao(self.output_line, self.trigger_line,
                                 self.run_line, self.fs, self.expected_range)
        self._tasks.extend((self._task_digital, self._task_analog))
        self.samples_written = 0

    def get_write_size(self):
        # We don't want the buffering algorithm to get too far ahead of us
        if self._state == 'halted':
            return 0
        if self.samples_written == 0:
            return self.write_size
        result = ctypes.c_uint64()
        ni.DAQmxGetWriteTotalSampPerChanGenerated(self._task_analog, result)
        buffered = self.samples_written-result.value
        write_size = self.write_size-buffered
        log.debug('%d samples transferred to buffer, '
                  '%d samples transferred to device '
                  '%d samples to add to buffer',
                  self.samples_written, result.value, write_size)
        return write_size

    def write(self, attenuation, analog, trigger, running):
        analog_result = ctypes.c_long()
        digital_result = ctypes.c_long()
        ni.DAQmxWriteAnalogF64(self._task_analog, analog.shape[-1], False, -1,
                               ni.DAQmx_Val_GroupByChannel,
                               analog.ravel().astype(np.double), analog_result,
                               None)
        data = np.r_[trigger, running]
        ni.DAQmxWriteDigitalLines(self._task_digital, len(trigger), False, -1,
                                  ni.DAQmx_Val_GroupByChannel,
                                  data.astype(np.uint8), digital_result, None)
        log.debug('Wrote %d samples to %s', analog_result.value,
                  self.output_line)
        log.debug('Wrote %d samples to %s', digital_result.value,
                  self.trigger_line)
        self.samples_written += analog_result.value

    def status(self):
        try:
            result = ctypes.c_ulong()
            ni.DAQmxIsTaskDone(self._task_analog, result)
        except:
            return self.samples_written
        result = ctypes.c_uint64()
        ni.DAQmxGetWriteTotalSampPerChanGenerated(self._task_analog, result)
        return result.value

    def complete(self):
        return self.status() == self.samples_written

    def get_nearest_atten(self, atten):
        if self.attenuator is None:
            raise ValueError('Cannot control attenuation')
        return self.attenuator.get_nearest_atten(atten)


class DAQmxAttenControl(DAQmxBase):
    '''
    Handles configuration and control of niDAQmx for controlling volume via
    serial control of a programmable IC chip
    '''
    # Gain control settings in dB.  This is IC specific.  Maximum volume is
    # +31.5 dB; however, I prefer not to use the gain stage since it makes it
    # difficult to calculate the maximum allowable voltage output from the NI
    # DAC (the NI DAC can provide sufficient output to cover the full range of
    # the BUF634 when the volume control is set to 0 dB).
    VOLUME_STEP = 0.5
    VOLUME_MAX = 31.5
    VOLUME_MIN = -96
    VOLUME_BITS = 16

    MAX_ATTEN = -VOLUME_MIN
    MIN_ATTEN = -VOLUME_MAX
    ATTEN_STEP = VOLUME_STEP

    # This is limited by the input characteristics of the IC (right now the
    # chip needs at least 90us from the time the chip select line goes low to
    # reading the first sample from the serial input.  In addition, it needs a
    # 20us set-up time for reading the next sample (i.e. the clock line must
    # not go high until 20us after the sample line goes high, and then it must
    # hold the sample line high for at least 20us).  This places an upper bound
    # of 10e6 on the clock frequency.  However, a more reasonable upper bound
    # is probably 1e6 (1GHz).  Minimum frequency is 6250 (100 kHz timebase/16).
    #VOLUME_CLOCK_FREQ = 1e5
    VOLUME_CLOCK_FREQ = 1e4

    # Each bit is represented by two samples in the buffer to allow the serial
    # clock to transition while the serial data line is held constant at the
    # control value.  Two additional bits are added (one at the beginning and
    # end of the signal) to bring the chip select line back to high.
    # Samp     1   2   3
    # Value    1   0   1
    # P0.0  1 0 0 0 0 0 0 1 - IC reads serial data when 0
    # P0.1  0 0 1 0 1 0 1 1 - IC reads a bit from P0.2 when 0 -> 1
    # P0.2  0 1 1 0 0 1 1 0 - Hold value at 0 or 1 when P0.1 0 -> 1
    VOLUME_BUFFER = VOLUME_BITS * 4 + 2

    fs = 200e3

    def __init__(self,
                 clock_line=DAQmxDefaults.VOLUME_CLK,
                 cs_line=DAQmxDefaults.VOLUME_CS,
                 data_line=DAQmxDefaults.VOLUME_SDI,
                 mute_line=DAQmxDefaults.VOLUME_MUTE,
                 zc_line=DAQmxDefaults.VOLUME_ZC,
                 hw_clock=DAQmxDefaults.DIO_CLOCK,
                 ):
        '''
        Parameters
        ----------
        clock_line : str
            Line to use for serial clock (rising edge of clock tells IC to read
            the state of the data line)
        cs_line : str
            Line to use for chip select (low value indicates that data is being
            written).
        data_line : str
            Line to use for serial data.
        mute_line : str
            Line to use for controlling mute state.  IC is muted when line is
            high.
        zc_line : str
            Line to use for controlling zero crossing state.  When high, IC
            will change attenuation only when a zero crossing of the signal
            occurs.
        hw_clock : str
            Hardware clock to use for controlling hardware-timed write of the
            serial lines (clock, cs, and data).  Important because the serial
            lines have very precise timing requirements.
        '''
        # Can be set to '/Dev1/FreqOut' or '/Dev1/CtrN' where N is a counter
        # number.  If we are using another sample clock (e.g. ao or ai), then
        # this will not be configured.
        self.serial_line = (cs_line, clock_line, data_line)
        self.mute_line = mute_line
        self.zc_line = zc_line
        self.hw_clock = hw_clock
        self._right_setting = None
        self._left_setting = None
        super(DAQmxAttenControl, self).__init__()

    def _setup_serial_timer(self, timer):
        # Configure timing of digital output using a counter that runs at twice
        # the frequency of the sample clock (enables us to set the value of the
        # serial data line high or low one cycle before setting the serial
        # clock high.  Here, the serial clock basically runs at half the rate
        # of the sample clock.
        task = create_task('COM timer')
        ni.DAQmxCreateCOPulseChanFreq(task, timer, '', ni.DAQmx_Val_Hz,
                                      ni.DAQmx_Val_Low, 0,
                                      self.VOLUME_CLOCK_FREQ, 0.5)
        ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)
        return task

    def _setup_serial_comm(self, volume_lines, timer=None):
        task = create_task('Serial COM')
        # Configure the task to send serial data to the PGA2311 (stereo audio
        # volume control).  This output should be fed into pin 6 (SCLK), P0.0
        # is chip select (CS) input.  The IC will only accept data when CS is
        # low.  P0.1 is the serial clock (feed into pin 6 of the IC, SCLK).
        # P0.2 is the serial data (16 bytes, MSB first, first 8 are right
        # attenuation, second 8 are left attenuation).
        ni.DAQmxCreateDOChan(task, ', '.join(volume_lines), '',
                             ni.DAQmx_Val_ChanPerLine)

        # Ensure that chip select line starts high to prevent writing to the
        # chip.  This needs to be done before configuring the implicit timing
        # or we'll get an error.
        initial = np.array([1, 0, 0], dtype=np.uint8)
        result = ctypes.c_int32()
        ni.DAQmxWriteDigitalLines(task, 1, True, -1,
                                  ni.DAQmx_Val_GroupByChannel, initial, result,
                                  None)
        ni.DAQmxWaitUntilTaskDone(task, -1)

        # Determine the internal timer to use for implicit timing
        if timer is not None:
            # If timer is not SampleClock, we need to set it up.
            if not timer.endswith('SampleClock'):
                task_clk = self._setup_serial_timer(timer)
            else:
                task_clk = None
            if timer.endswith('SampleClock'):
                internal_clock = timer
            elif timer.endswith('FreqOut'):
                internal_clock = '/Dev1/FrequencyOutput'
            else:
                internal_clock = timer + 'InternalOutput'
            ni.DAQmxCfgSampClkTiming(task, internal_clock, self.fs,
                                     ni.DAQmx_Val_Rising,
                                     ni.DAQmx_Val_FiniteSamps,
                                     self.VOLUME_BUFFER)
        else:
            task_clk = None

        ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)
        return task, task_clk

    def setup(self):
        # Configure the tasks and IO lines
        log.debug('Configuring NI tasks')
        self._task_mute = create_task('Mute control')
        self._task_zc = create_task('ZC control')

        self._task_com, self._task_clk = \
            self._setup_serial_comm(self.serial_line, self.hw_clock)

        # Set up mute line.  In theory we can combine both of these lines into a
        # single task, but it's easier to program them separately.
        ni.DAQmxCreateDOChan(self._task_mute, self.mute_line, 'mute',
                             ni.DAQmx_Val_ChanPerLine)

        # Set up zero crossing line
        ni.DAQmxCreateDOChan(self._task_zc, self.zc_line, 'zc',
                             ni.DAQmx_Val_ChanPerLine)

        if self._task_clk is not None:
            ni.DAQmxStartTask(self._task_clk)

        # Use the soft mute option
        self.set_mute(False)
        self.set_zero_crossing(False)
        self.set_atten(np.inf, np.inf)

        self._tasks.extend((self._task_clk, self._task_zc, self._task_mute,
                            self._task_com))

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

    def _build_com_signal(self, bits):
        samples_per_bit = 4
        samples = self.VOLUME_BITS*samples_per_bit
        line_csi = np.zeros(samples, dtype=np.uint8)  # chip select line
        line_clk = np.zeros(samples, dtype=np.uint8)  # sample clock line
        line_sdi = np.zeros(samples, dtype=np.uint8)  # sample data line
        line_clk[1::samples_per_bit] = 1
        for i, bit in enumerate(bits):
            offset = i*samples_per_bit
            line_sdi[offset:offset+samples_per_bit] = bit
        line_csi = np.pad(line_csi, 1, mode='constant', constant_values=1)
        line_clk = np.pad(line_clk, 1, mode='constant', constant_values=0)
        line_sdi = np.pad(line_sdi, 1, mode='constant', constant_values=0)
        return np.vstack((line_csi, line_clk, line_sdi))

    def _send_bits(self, bits):
        com_signal = self._build_com_signal(bits)
        samples = com_signal.shape[-1]
        result = ctypes.c_int32()
        ni.DAQmxWriteDigitalLines(self._task_com, samples, False,
                                  -1, ni.DAQmx_Val_GroupByChannel,
                                  com_signal.ravel().astype(np.uint8), result,
                                  None)
        ni.DAQmxStartTask(self._task_com)
        ni.DAQmxWaitUntilTaskDone(self._task_com, 1)
        ni.DAQmxStopTask(self._task_com)

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
        print right, left
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

    def set_attens(self, attens):
        '''
        Utility function for setting atten of right and left to same value

        This is a convenience method that allows one to specify the volume as an
        attenuation rather than a gain as it passes the inverted values to
        `set_gains`.
        '''
        self.set_gains([-a for a in attens])

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

    def get_nearest_atten(self, atten):
        hw_atten = np.floor(atten/self.ATTEN_STEP)*self.ATTEN_STEP
        return np.clip(hw_atten, self.MIN_ATTEN, self.MAX_ATTEN)


class DAQmxAcquire(object):

    def __init__(self, signal, repetitions, output_line, input_line,
                 gain, calibration, dac_fs=100e3, adc_fs=100e3, duration=1,
                 iti=0.01, callback=None):

        for k, v in locals().items():
            setattr(self, k, v)

        self.complete = False
        self.waveforms = []
        self.epochs_acquired = 0

        # Set up the recording
        self.iface_adc = TriggeredDAQmxSource(fs=adc_fs,
                                              epoch_duration=duration,
                                              input_line=input_line,
                                              callback=self.poll)
        self.iface_adc.setup()
        self.iface_adc.start()

        self.iface_atten = DAQmxAttenControl()
        self.iface_atten.setup()
        self.iface_atten.set_gain(gain)
        self.iface_atten.clear()

        channel = DAQmxChannel(token=signal, calibration=calibration,
                               output_line=output_line, duration=duration)
        self.iface_dac = DAQmxPlayer(fs=dac_fs, duration=duration)
        self.iface_dac.add_channel(channel)
        self.iface_dac.queue_init('FIFO')
        self.iface_dac.queue_append(repetitions, iti)

    def start(self):
        self.iface_dac.play_queue()

    def stop(self):
        self.iface_adc.clear()
        self.iface_dac.clear()

    def join(self):
        while not self.complete:
            time.sleep(0.1)

    def poll(self):
        waveforms = self.iface_adc.read_analog()
        self.epochs_acquired += len(waveforms)
        self.waveforms.append(waveforms)
        if self.epochs_acquired >= self.repetitions:
            self.stop()
            self.waveforms = np.concatenate(self.waveforms, axis=0)
            self.complete = True
        if self.callback is not None:
            self.callback(self.epochs_acquired, self.complete)


def acquire(*args, **kwargs):
    daq = DAQmxAcquire(*args, **kwargs)
    daq.start()
    daq.join()
    return daq.waveforms


################################################################################
# Tests
################################################################################
class TestDAQmxAttenControl(unittest.TestCase):

    def test_bitword(self):
        test_values = [
            ((254, 8, 'big-endian'), [1, 1, 1, 1, 1, 1, 1, 0]),
            ((254, 8, 'little-endian'), [0, 1, 1, 1, 1, 1, 1, 1]),
            ((128, 8, 'little-endian'), [0, 0, 0, 0, 0, 0, 0, 1]),
            ((128, 8, 'big-endian'), [1, 0, 0, 0, 0, 0, 0, 0]),
            ((128, 16, 'big-endian'), [0, 0, 0, 0, 0, 0, 0, 0,
                                       1, 0, 0, 0, 0, 0, 0, 0]),
            ((128, 16, 'big-endian'), [0, 0, 0, 0, 0, 0, 0, 0,
                                       1, 0, 0, 0, 0, 0, 0, 0]),
            ((254 | (128 << 8), 16, 'big-endian'), [1, 0, 0, 0, 0, 0, 0, 0,
                                                    1, 1, 1, 1, 1, 1, 1, 0]),
            ((254 | (128 << 8), 16, 'little-endian'), [0, 1, 1, 1, 1, 1, 1, 1,
                                                       0, 0, 0, 0, 0, 0, 0, 1]),
        ]
        for args, expected in test_values:
            actual = get_bits(*args)
            self.assertEqual(actual, expected)

    def test_gain_to_bits(self):
        atten = DAQmxAttenControl()
        self.assertEqual(atten._gain_to_byte(31.5), 255)
        self.assertEqual(atten._gain_to_byte(-95.5), 1)

        # Right, left
        actual = atten._gain_to_bits(31.0, -32.0)
        expected = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
