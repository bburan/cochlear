'''
Classes for configuring and recording using NIDAQmx compatible devices
'''

# TODO setup DAQmxRegisterDoneEvent callback
# TODO automate registry of pointer callbacks to keep them from getting
# garbage-collected

from __future__ import division

import ctypes
import unittest
import time

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
    DEV = 'Dev1'

    # See documentation in nidaqmx.TriggeredDAQmxSource for information on the
    # specifics of each line.
    ERP_INPUT = '/{}/ai3'.format(DEV)
    MIC_INPUT = '/{}/ai1'.format(DEV)
    REF_MIC_INPUT = '/{}/ai2'.format(DEV)

    AI_COUNTER = '/{}/Ctr0'.format(DEV)
    AI_TRIGGER = '/{}/PFI0'.format(DEV)
    AI_RUN = None
    AI_RANGE = 10

    # Digital output used to indicate start of trial. Must support
    # hardware-timed DIO.
    AO_TRIGGER = '/{}/port0/line0'.format(DEV)
    # Digital output used to indicate that a stimulus is being played out (will
    # be high for full sequence of trials). Must support hardware-timed DIO.
    AO_RUN = '/{}/port0/line1'.format(DEV)

    AO_RANGE = np.sqrt(2)
    DUAL_SPEAKER_OUTPUT = '/{}/ao0:1'.format(DEV)
    PRIMARY_SPEAKER_OUTPUT = '/{}/ao0'.format(DEV)
    SECONDARY_SPEAKER_OUTPUT = '/{}/ao1'.format(DEV)
    AO0_ATTEN_CHANNEL = 'left'
    AO1_ATTEN_CHANNEL = 'right'
    PRIMARY_ATTEN_CHANNEL = AO0_ATTEN_CHANNEL
    SECONDARY_ATTEN_CHANNEL = AO1_ATTEN_CHANNEL

    AI_FS = 200e3
    AO_FS = 200e3

    # Will be software-timed DIO
    VOLUME_CLK = '/{}/port1/line2'.format(DEV)
    VOLUME_CS = '/{}/port1/line3'.format(DEV)
    VOLUME_SDI = '/{}/port1/line4'.format(DEV)
    VOLUME_MUTE = '/{}/port1/line5'.format(DEV)
    VOLUME_ZC = '/{}/port1/line6'.format(DEV)

    TRIGGER_DURATION = 1e-3


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


def create_continuous_ai(ai, fs, run_line=None, expected_range=10, callback=None,
                         callback_samples=None, delay_samples=0,
                         posttrigger_samples=2, task=None):
    '''
    Parameters
    ----------
    ai : str
        Analog input line(s) to acquire in this task
    run_line : str
        Digital input line.  Rising edge triggers acquisition, falling edge
        halts acquisition.
    expected_range : float
        Range of signal (in volts).
    callback : callable
        Function to call with acquired data
    callback_samples : int
    '''
    if task is None:
        task = create_task()
    vlb, vub = -expected_range, expected_range
    ni.DAQmxCreateAIVoltageChan(task, ai, '', ni.DAQmx_Val_Diff, vlb,
                                vub, ni.DAQmx_Val_Volts, '')

    # Set up acquisition start/end trigger.  The acquisition will continue after
    # the stop trigger by the specified number of posttrigger samples.  We need
    # to set the buffer read relative to "current read position" to enable the
    # DAQmx read functions to read samples that are put into the buffer after
    # the start trigger occurs (just like a normal continuous read).  If we
    # don't change this property, then the read would wait until the reference
    # trigger was recieved before it could retrieve samples from the buffer.
    # Also important, the buffer size must be explicitly set to a relatively
    # large size otherwise it will be set at posttrigger_samples+2.  Data would
    # almost certainly be lost under this scenario!
    if run_line is not None:
        ni.DAQmxCfgDigEdgeStartTrig(task, run_line, ni.DAQmx_Val_Rising)
        ni.DAQmxCfgDigEdgeRefTrig(task, run_line, ni.DAQmx_Val_Falling, 2)
        ni.DAQmxCfgSampClkTiming(task, None, fs, ni.DAQmx_Val_Rising,
                                ni.DAQmx_Val_FiniteSamps, posttrigger_samples+2)
        ni.DAQmxSetReadRelativeTo(task, ni.DAQmx_Val_CurrReadPos)
        ni.DAQmxSetBufInputBufSize(task, int(callback_samples*1000))
    else:
        ni.DAQmxCfgSampClkTiming(task, None, fs, ni.DAQmx_Val_Rising,
                                 ni.DAQmx_Val_ContSamps,
                                 int(callback_samples*1000))

    # Delay acquisition relative to the start trigger.
    if delay_samples != 0:
        ni.DAQmxSetStartTrigDelayUnits(task, ni.DAQmx_Val_SampClkPeriods)
        ni.DAQmxSetStartTrigDelay(task, delay_samples)

    ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)
    if callback is not None:
        cb_ptr = create_everynsamples_callback(callback, callback_samples,
                                               task)
    else:
        cb_ptr = None
    return task, cb_ptr


def create_retriggerable_ai(ai, fs, epoch_size, trigger, expected_range=10,
                            counter=None, run=None,
                            record_mode=ni.DAQmx_Val_Diff, callback=None,
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
    record_mode : pass

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
    ni.DAQmxCreateAIVoltageChan(task_analog, ai, '', record_mode, vlb, vub,
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


def create_ao(ao, trigger, run, fs, expected_range=DAQmxDefaults.AO_RANGE,
              callback=None, callback_samples=None, duration=None,
              done_callback=None, allow_regen=False):

    task_analog = create_task()
    task_digital = create_task()
    vmin, vmax = -expected_range, expected_range

    regen_mode = ni.DAQmx_Val_AllowRegen if allow_regen \
        else ni.DAQmx_Val_DoNotAllowRegen

    # Setup analog output and prevent playout of data that has already been
    # played.
    ni.DAQmxCreateAOVoltageChan(task_analog, ao, '', vmin, vmax,
                                ni.DAQmx_Val_Volts, '')
    ni.DAQmxSetWriteRegenMode(task_analog, regen_mode)

    if duration is None:
        print 'duration is None'
        ni.DAQmxCfgSampClkTiming(task_analog, '', fs, ni.DAQmx_Val_Rising,
                                 ni.DAQmx_Val_ContSamps, int(fs))
        output_buffer_size = int(fs*10)
        ni.DAQmxCfgOutputBuffer(task_analog, int(fs*10))
        log.debug('Configured continuous output with output buffer size %d',
                  output_buffer_size)
    else:
        output_samples = int(duration*fs)
        log.debug('Configured finite output with %d samples', output_samples)
        ni.DAQmxCfgSampClkTiming(task_analog, '', fs, ni.DAQmx_Val_Rising,
                                 ni.DAQmx_Val_FiniteSamps, output_samples)

    # Set up trigger line, ensuring that trigger and run lines are at LOW (in
    # case previous task aborted before the zero samples were written out).
    do = ','.join([trigger, run])
    ni.DAQmxCreateDOChan(task_digital, do, '', ni.DAQmx_Val_ChanPerLine)
    ni.DAQmxWriteDigitalLines(task_digital, 1, True, -1,
                              ni.DAQmx_Val_GroupByChannel,
                              np.array([0, 0], dtype=np.uint8),
                              ctypes.c_int32(), None)
    ni.DAQmxSetWriteRegenMode(task_digital, regen_mode)
    ni.DAQmxCfgSampClkTiming(task_digital, 'ao/SampleClock', fs,
                             ni.DAQmx_Val_Rising, ni.DAQmx_Val_ContSamps,
                             int(fs))

    if callback is not None:
        cb_ptr = create_everynsamples_callback(callback, callback_samples,
                                               task_analog, 'output')
    else:
        cb_ptr = None

    if done_callback is not None:
        done_cb_ptr = create_done_callback(done_callback, task_analog)
    else:
        done_cb_ptr = None

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

    return task_analog, task_digital, cb_ptr, done_cb_ptr


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
        log.debug('Exiting anonymous every N samples callback')
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


class DAQmxSource(DAQmxBase):
    '''
    Read data from device
    '''

    DIFF = ni.DAQmx_Val_Diff
    NRSE = ni.DAQmx_Val_NRSE
    RSE = ni.DAQmx_Val_RSE

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

    def setup(self):
        super(DAQmxSource, self).setup()
        self._event = threading.Event()

    def start(self):
        super(DAQmxSource, self).start()
        self._event.clear()


class ContinuousDAQmxSource(DAQmxSource):

    def __init__(self, fs=25e3, input_line='/Dev1/ai0', callback=None,
                 callback_samples=None, expected_range=10, run_line=None,
                 delay_samples=0, pipeline=None, complete_callback=None):
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
        for k, v in locals().items():
            setattr(self, k, v)
        super(ContinuousDAQmxSource, self).__init__()

    def setup(self):
        log.debug('Setting up continuous AI tasks')
        self._task_analog, self._cb_ptr = create_continuous_ai(
            ai=self.input_line,
            fs=self.fs,
            run_line=self.run_line,
            expected_range=self.expected_range,
            callback=self.trigger_callback,
            callback_samples=self.callback_samples,
            delay_samples=self.delay_samples,
        )
        self._tasks.append(self._task_analog)
        self.channels = num_channels(self._task_analog)
        log.debug('Configured continous AI with %d channels controlled by %s',
                  self.channels, self.run_line)
        super(ContinuousDAQmxSource, self).setup()

    def samples_available(self):
        return samples_available(self._task_analog)


class TriggeredDAQmxSource(DAQmxSource):

    def __init__(self, input_line, fs, epoch_duration,
                 trigger_line=DAQmxDefaults.AI_TRIGGER,
                 expected_range=DAQmxDefaults.AI_RANGE,
                 counter_line=DAQmxDefaults.AI_COUNTER,
                 run_line=DAQmxDefaults.AI_RUN,
                 trigger_duration=DAQmxDefaults.TRIGGER_DURATION,
                 trigger_delay=0, callback=None, pipeline=None,
                 complete_callback=None, record_mode=DAQmxSource.DIFF):
        '''
        Parameters
        ----------
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
        if callback is None and pipeline is None:
            raise ValueError('Must provide callback or pipeline')
        for k, v in locals().items():
            setattr(self, k, v)
        self.offset = int(trigger_delay * fs)
        self.epoch_size = int(epoch_duration * fs) + self.offset
        super(TriggeredDAQmxSource, self).__init__()

    def setup(self):
        log.debug('Setting up triggered AI tasks')
        self._task_analog, self._task_digital, self._cb_ptr = \
            create_retriggerable_ai(self.input_line, self.fs, self.epoch_size,
                                    self.trigger_line, self.expected_range,
                                    self.counter_line, self.run_line,
                                    self.record_mode, self.trigger_callback)
        self._tasks.extend((self._task_analog, self._task_digital))
        self.channels = num_channels(self._task_analog)
        log.debug('Configured retriggerable AI with %d channels', self.channels)
        super(TriggeredDAQmxSource, self).setup()

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
        return analog_data.swapaxes(0, 1)[:, :, self.offset:]


class DAQmxChannel(Channel):

    # Based on testing of the PGA2310/OPA234/BUF634 circuit with volume control
    # set to 0 dB, this is the maximum allowable voltage without clipping.
    #voltage_min = -1.1*np.sqrt(2)
    #voltage_max = 1.1*np.sqrt(2)
    voltage_min = -2.5*np.sqrt(2)
    voltage_max = 2.5*np.sqrt(2)

    attenuator = ScalarInput(default=None, unit=None)
    attenuator_channel = ScalarInput(default=0, unit=None)

    def set_gain(self, gain):
        return self.set_attenuation(-gain)

    def set_attenuation(self, atten):
        self.attenuator.set_atten(**{self.attenuator_channel: atten})
        self.attenuation = atten

    def get_nearest_attenuation(self, atten):
        if self.attenuator is None:
            raise ValueError('Channel does not have attenuation control')
        return self.attenuator.get_nearest_atten(atten)


class DAQmxPlayer(DAQmxBase):

    def __init__(self, fs=200e3,
                 output_line=DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
                 expected_range=DAQmxDefaults.AO_RANGE,
                 trigger_line=DAQmxDefaults.AO_TRIGGER,
                 run_line=DAQmxDefaults.AO_RUN, attenuator=None,
                 buffer_size=2, monitor_interval=0.5, total_duration=None,
                 done_callback=None, allow_regen=False, **kwargs):
        '''
        Parameters
        ----------
        buffer_size : float, sec
            Number of samples available to niDAQmx for playout.  The larger this
            value, the longer it takes for changes to be applied.  Smaller
            values have a greater risk of not keeping up with playout.
        monitor_interval : float, sec
            How often should new samples be uploaded to the buffer?
        '''
        variables = locals()
        kwargs = variables.pop('kwargs')
        for k, v in variables.items():
            setattr(self, k, v)
        super(DAQmxPlayer, self).__init__(**kwargs)

    def setup(self):
        log.debug('Setting up AO tasks')
        self.buffer_samples = int(self.fs*self.buffer_size)
        self.monitor_samples = int(self.fs*self.monitor_interval)
        log.debug('Buffer size %d, monitor every %d samples',
                  self.buffer_samples, self.monitor_samples)

        kwargs = {}
        if hasattr(self, 'callback') and self.monitor_interval is not None:
            kwargs['callback'] = self.callback
            kwargs['callback_samples'] = self.callback_samples
        kwargs['allow_regen'] = getattr(self, 'allow_regen', False)

        self._task_analog, self._task_digital, self._cb_ptr, \
            self._done_cb_ptr = create_ao(
                self.output_line,
                self.trigger_line,
                self.run_line,
                self.fs,
                self.expected_range,
                duration=self.total_duration,
                done_callback=self._done_callback,
                **kwargs)

        # Order is important.  Digital task needs to be started before analog
        # task.
        self._tasks.extend((self._task_digital, self._task_analog))
        self.samples_written = 0
        self.nchannels = num_channels(self._task_analog)
        log.debug('Configured AO with %d channels', self.nchannels)
        super(DAQmxPlayer, self).setup()

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

    def simple_write(self, analog, iti,
                     trigger_duration=DAQmxDefaults.TRIGGER_DURATION):
        analog, trigger, running = prepare_for_write(analog, self.fs, iti,
                                                     trigger_duration)
        return self.write(analog, trigger, running)

    def write(self, analog, trigger, running):
        analog = np.asarray(analog)
        trigger = np.asarray(trigger)
        running = np.asarray(running)

        analog_result = ctypes.c_long()
        digital_result = ctypes.c_long()
        log.debug('Writing array of shape %r to AO', analog.shape)
        log.debug('First sample %f, last sample %f', analog.ravel()[0],
                  analog.ravel()[-1])
        ni.DAQmxWriteAnalogF64(self._task_analog, analog.shape[-1], False, -1,
                               ni.DAQmx_Val_GroupByChannel,
                               analog.ravel().astype(np.double), analog_result,
                               None)
        data = np.r_[trigger, running]
        ni.DAQmxWriteDigitalLines(self._task_digital, len(trigger), False, -1,
                                  ni.DAQmx_Val_GroupByChannel,
                                  data.astype(np.uint8), digital_result, None)
        log.debug('Wrote %d s/chan to %s', analog_result.value,
                  self.output_line)
        log.debug('Wrote %d s/chan to %s and %s', digital_result.value,
                  self.trigger_line, self.run_line)
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

    def get_nearest_atten(self, atten):
        if self.attenuator is None:
            raise ValueError('Cannot control attenuation')
        return self.attenuator.get_nearest_atten(atten)

    def _done_callback(self):
        if self.done_callback is not None:
            self.done_callback()


class QueuedDAQmxPlayer(DAQmxPlayer, QueuedPlayer):

    pass


class ContinuousDAQmxPlayer(DAQmxPlayer, ContinuousPlayer):

    def clear(self):
        super(ContinuousDAQmxPlayer, self).clear()
        self.reset()


class DAQmxOutput(DAQmxBase):

    def __init__(self, fs=200e3,
                 output_line=DAQmxDefaults.PRIMARY_SPEAKER_OUTPUT,
                 expected_range=DAQmxDefaults.AO_RANGE,
                 trigger_line=DAQmxDefaults.AO_TRIGGER,
                 run_line=DAQmxDefaults.AO_RUN, attenuator=None,
                 buffer_size=2, monitor_interval=0.5, total_duration=None,
                 done_callback=None, **kwargs):
        '''
        Parameters
        ----------
        buffer_size : float, sec
            Number of samples available to niDAQmx for playout.  The larger this
            value, the longer it takes for changes to be applied.  Smaller
            values have a greater risk of not keeping up with playout.
        monitor_interval : float, sec
            How often should new samples be uploaded to the buffer?
        '''
        variables = locals()
        kwargs = variables.pop('kwargs')
        for k, v in variables.items():
            setattr(self, k, v)
        super(DAQmxPlayer, self).__init__(**kwargs)


class DAQmxAttenControl(DAQmxBase):
    '''
    Handles configuration and control of niDAQmx for controlling volume via
    serial control of a programmable IC chip
    '''
    # Gain control settings in dB.  This is IC specific.  Maximum volume is
    # +31.5 dB.
    VOLUME_STEP = 0.5
    VOLUME_MAX = 31.5
    #VOLUME_MIN = -96
    VOLUME_MIN = -50
    #VOLUME_MAX = 0
    VOLUME_BITS = 16

    MAX_ATTEN = -VOLUME_MIN
    MIN_ATTEN = -VOLUME_MAX
    ATTEN_STEP = VOLUME_STEP

    fs = 200e3

    def __init__(self,
                 clk_line=DAQmxDefaults.VOLUME_CLK,
                 cs_line=DAQmxDefaults.VOLUME_CS,
                 sdi_line=DAQmxDefaults.VOLUME_SDI,
                 mute_line=DAQmxDefaults.VOLUME_MUTE,
                 zc_line=DAQmxDefaults.VOLUME_ZC,
                 ):
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
        super(DAQmxAttenControl, self).__init__()

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
        super(DAQmxAttenControl, self).setup()

    def start(self):
        super(DAQmxAttenControl, self).start()
        self._send_bit(self._task_cs, 1)
        self.set_mute(False)
        self.set_zero_crossing(False)
        self.set_atten(np.inf, np.inf)

    def clear(self):
        for task in (self._task_zc, self._task_mute, self._task_clk,
                     self._task_cs, self._task_sdi):
            ni.DAQmxClearTask(task)
        super(DAQmxAttenControl, self).clear()

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


################################################################################
# Primary helpers for DAQ
################################################################################
class DAQmxAcquire(object):

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
        self.iface_adc = TriggeredDAQmxSource(fs=adc_fs,
                                              epoch_duration=duration,
                                              input_line=input_line,
                                              expected_range=input_range,
                                              callback=self.poll)
        self.iface_adc.start()

        self.iface_atten = DAQmxAttenControl()
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

    def stop(self):
        self.iface_adc.clear()
        self.iface_dac.clear()

    def join(self):
        while not self.complete:
            time.sleep(0.1)

    def poll(self, waveforms):
        log.debug('Received data')
        lb, ub = self.epochs_acquired, self.epochs_acquired+len(waveforms)
        self.waveform_buffer[lb:ub, :, :] = waveforms
        self.epochs_acquired += len(waveforms)
        if self.epochs_acquired >= self.repetitions:
            log.debug('Halting acquisition')
            self.stop()
            self.complete = True
        if self.callback is not None:
            self.callback(self.epochs_acquired, self.complete)
        log.debug('Completed processing received data')


def acquire(*args, **kwargs):
    daq = DAQmxAcquire(*args, **kwargs)
    daq.start()
    daq.join()
    return daq.waveform_buffer


class SimpleDAQmxAcquire(object):

    def __init__(self, waveform, repetitions, output_line, input_line, gain,
                 dac_fs=100e3, adc_fs=100e3, iti=0.01, input_range=10,
                 callback=None):

        log.debug('Setting up acquisition')
        for k, v in locals().items():
            setattr(self, k, v)

        self.complete = False
        self.waveforms = []
        self.epochs_acquired = 0

        waveform_duration = waveform.shape[-1]/dac_fs
        trial_duration = waveform_duration+iti
        total_duration = trial_duration*repetitions
        epoch_duration = trial_duration-adc_fs**-1

        # Set up the recording
        self.iface_adc = TriggeredDAQmxSource(fs=adc_fs,
                                              epoch_duration=epoch_duration,
                                              input_line=input_line,
                                              expected_range=input_range,
                                              callback=self.poll)
        self.iface_atten = DAQmxAttenControl()
        self.iface_atten.setup(gain)

        self.iface_dac = DAQmxPlayer(fs=dac_fs, total_duration=total_duration,
                                     output_line=output_line, allow_regen=True)
        self.iface_dac.setup()
        self.iface_dac.simple_write(waveform, iti)

    def start(self):
        log.debug('Starting acquisition')
        self.iface_adc.start()
        self.iface_dac.start()

    def stop(self):
        self.iface_adc.clear()
        self.iface_dac.clear()

    def join(self):
        while not self.complete:
            time.sleep(0.1)

    def poll(self, waveforms):
        log.debug('Received data')
        lb, ub = self.epochs_acquired, self.epochs_acquired+len(waveforms)
        print waveforms.shape
        self.waveforms.append(waveforms)
        self.epochs_acquired += len(waveforms)
        if self.epochs_acquired >= self.repetitions:
            log.debug('Halting acquisition')
            self.stop()
            self.complete = True
        if self.callback is not None:
            self.callback(self.epochs_acquired, self.complete)
        log.debug('Completed processing received data')


def simple_acquire(waveform, fs, repetitions, input_line, iti,
                   trigger_duration=0.01, epoch_size=None):

    waveform_duration = len(waveform)/fs
    trial_duration = waveform_duration+iti
    total_duration = trial_duration*repetitions

    # Set epoch size to just smaller than the trial duration to ensure that the
    # trigger has time to rearm before the next acquisition.
    if epoch_size is None:
        epoch_size = trial_duration-0.001

    iti_samples = np.int(fs*iti)
    trigger_samples = np.int(fs*trigger_duration)

    running = np.ones_like(waveform)
    trigger = np.zeros_like(waveform)
    trigger[:trigger_samples] = 1

    waveform = np.pad(waveform, (0, iti_samples), 'constant')
    running = np.pad(running, (0, iti_samples), 'constant')
    trigger = np.pad(trigger, (0, iti_samples), 'constant')

    def poll(waveforms):
        print 'acquired', waveforms.shape

    acq = TriggeredDAQmxSource(fs=fs, epoch_duration=epoch_size,
                               input_line='/Dev1/ai1', callback=poll)
    player = DAQmxPlayer(fs=fs, total_duration=total_duration, allow_regen=True)
    player.setup()
    acq.setup()
    player.write(waveform, trigger, running)
    acq.start()
    player.start()
    while True:
        import time
        time.sleep(0.5)


def main_golay():
    from neurogen.calibration.util import golay_pair, golay_transfer_function
    a, b = golay_pair(16)
    fs = 200e3
    kwargs = dict(repetitions=10, output_line='/Dev1/ao0',
                  input_line=DAQmxDefaults.MIC_INPUT, gain=0, dac_fs=fs,
                  adc_fs=fs, iti=0.001)

    acq = SimpleDAQmxAcquire(a, **kwargs)
    acq.start()
    acq.join()
    a_resp = np.concatenate(acq.waveforms).mean(axis=0)[0]

    acq = SimpleDAQmxAcquire(b, **kwargs)
    acq.start()
    acq.join()
    b_resp = np.concatenate(acq.waveforms).mean(axis=0)[0]

    freq, psd = golay_transfer_function(a, b, a_resp, b_resp, fs)

    import pylab as pl
    pl.semilogx(freq, np.log10(psd))
    pl.axis(xmin=100, xmax=100000)
    pl.show()


def main_chirp():
    from neurogen.calibration.util import transfer_function
    import scipy.signal
    fs = 200e3
    t = np.arange(0.2*fs)/fs
    y = scipy.signal.chirp(t, 100, 0.2, 60e3)
    kwargs = dict(repetitions=10, output_line='/Dev1/ao0',
                  input_line=DAQmxDefaults.MIC_INPUT, gain=-10, dac_fs=fs,
                  adc_fs=fs, iti=0.001)

    acq = SimpleDAQmxAcquire(y, **kwargs)
    acq.start()
    acq.join()
    resp = np.concatenate(acq.waveforms).mean(axis=0)[0]

    freq, psd = transfer_function(y, resp, fs)

    import pylab as pl
    #pl.plot(t, y, 'k-')
    pl.semilogx(freq, np.log10(psd))
    pl.axis(xmin=100, xmax=100000)


if __name__ == '__main__':
    main_chirp()
    main_golay()
    pl.show()
