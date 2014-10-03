'''
Classes for configuring and recording using NIDAQmx compatible devices
'''

# TODO setup DAQmxRegisterDoneEvent callback
# TODO auto-reset devices on module load via DAQmxResetDevice?

from __future__ import division

import ctypes

import PyDAQmx as ni
import numpy as np

from neurogen.sink import Sink

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class DAQmxDefaults(object):
    '''
    Define a set of default lines that are typically hard-wired
    '''
    DEV = 'Dev1'

    # See documentation in nidaqmx.TriggeredDAQmxSource for information on the
    # specifics of each line.
    ERP_INPUT = '/{}/ai0'.format(DEV)
    ERP_COUNTER = '/{}/Ctr0'.format(DEV)
    ERP_TRIGGER = '/{}/PFI1'.format(DEV)

    MIC_INPUT = '/{}/ai1'.format(DEV)
    MIC_COUNTER = '/{}/Ctr1'.format(DEV)
    MIC_TRIGGER = '/{}/PFI2'.format(DEV)

    SPEAKER_OUTPUT = '/{}/ao0'.format(DEV)
    SPEAKER_TRIGGER = '/{}/port0/line0'.format(DEV)
    SPEAKER_RUN = '/{}/port0/line2'.format(DEV)

    VOLUME_CLK = '/{}/port0/line4'.format(DEV)
    VOLUME_CS = '/{}/port0/line1'.format(DEV)
    VOLUME_SDI = '/{}/port0/line6'.format(DEV)
    VOLUME_MUTE = '/{}/port1/line5'.format(DEV)
    VOLUME_ZC = '/{}/port1/line6'.format(DEV)
    DIO_CLOCK = '/{}/FreqOut'.format(DEV)


class DAQmxBase(object):

    def __init__(self):
        self._tasks = []
        self._state = 'initialized'

    def _create_task(self, name=None):
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

    def register_change_detect_callback(self, callback, rising=None,
                                        falling=None):
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
        if rising is None and falling is None:
            raise ValueError('Must provide at least one line')
        if rising is None:
            rising = ''
        if falling is None:
            falling = ''
        self._change_detect_callback = callback
        task = self._create_task()
        line = rising if rising == falling else ','.join((rising, falling))
        ni.DAQmxCreateDIChan(task, line, '', ni.DAQmx_Val_ChanForAllLines)
        ni.DAQmxCfgChangeDetectionTiming(task, rising, falling,
                                         ni.DAQmx_Val_ContSamps, 100)

        # Pointers to classmethods are not supported by ctypes, so we need
        # to take advantage of function closure to maintain a reference to
        # self.  Must return 0 to keep NIDAQmx happy (NIDAQmx expects all
        # functions to return 0 to indicate success.  A non-zero return
        # value indicates there was a nerror).
        def event_cb(task, signal_ID, data):
            self._change_detect_callback()
            return 0

        # Binding the pointer to an attribute on self seems to be necessary
        # to ensure the callback function does not disappear (presumably it
        # gets garbage-collected otherwise)
        self._change_detect_callback_ptr = \
            ni.DAQmxSignalEventCallbackPtr(event_cb)
        ni.DAQmxRegisterSignalEvent(task, ni.DAQmx_Val_ChangeDetectionEvent, 0,
                                    self._change_detect_callback_ptr, None)

        ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)
        self._task_change_detect = task
        self._tasks.append(task)

    def create_event_timer(self, trigger, counter='/Dev1/Ctr0',
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
            task = self._create_task()
        ni.DAQmxCreateCICountEdgesChan(task, counter, '', ni.DAQmx_Val_Rising,
                                       0, ni.DAQmx_Val_CountUp)
        ni.DAQmxSetCICountEdgesTerm(task, counter, clock)
        ni.DAQmxCfgSampClkTiming(task, trigger, self.fs, ni.DAQmx_Val_Rising,
                                 ni.DAQmx_Val_FiniteSamps, 500)
        ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)
        return task


class TriggeredDAQmxSource(DAQmxSource):
    '''
    Acquire epochs of data triggered off of an external signal

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
    used here may not be necessary.
    '''

    def __init__(self, fs=25e3, epoch_duration=10e-3, input_line='/Dev1/ai0',
                 counter_line='/Dev1/Ctr0', trigger_line='/Dev1/PFI1',
                 run_line=None, trigger_duration=1e-3, callback=None):
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
        self._task_analog, self._task_digital = \
            self.create_retriggerable_ai(self.input_line, self.trigger_line,
                                         self.counter_line, self.run_line,
                                         self.callback)
        self._tasks.extend((self._task_analog, self._task_digital))

        result = ctypes.c_uint32()
        ni.DAQmxGetTaskNumChans(self._task_analog, result)
        self.channels = result.value
        log.debug('Configured retriggerable AI with %d channels', self.channels)

    def create_retriggerable_ai(self, ai, trigger, counter=None, run=None,
                                callback=None, task_input=None,
                                task_record=None):
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
        task_input : {None, niDAQmx task}
            Task to use for reading analog data in.  If None, a new task will be
            created.
        task_record : {None, niDAQmx task}
            Task to use for controlling record counter.  If None, a new task
            will be created.

        Returns
        -------
        task_input
            Configured and committed task for analog data
        task_record
            Configured and committed task for triggering acquisition of analog
            data.
        '''
        if task_input is None:
            task_input = self._create_task()
        if task_record is None:
            task_record = self._create_task()

        # Set up the analog input to continously acquire samples.  However,
        # since it's tied to the internal output of the counter, it will only
        # acquire when the counter is active.  Since the counter only runs for a
        # finite number of samples after each trigger, this is effectively a
        # triggered
        ni.DAQmxCreateAIVoltageChan(task_input, ai, '', ni.DAQmx_Val_RSE, -10,
                                    10, ni.DAQmx_Val_Volts, '')
        ni.DAQmxCfgSampClkTiming(task_input, counter+'InternalOutput', self.fs,
                                 ni.DAQmx_Val_Rising, ni.DAQmx_Val_ContSamps,
                                 self.epoch_size)
        ni.DAQmxSetBufInputBufSize(task_input, self.epoch_size*1000)

        ni.DAQmxCreateCOPulseChanFreq(task_record, counter, '', ni.DAQmx_Val_Hz,
                                      ni.DAQmx_Val_Low, 0, self.fs, 0.5)
        ni.DAQmxCfgImplicitTiming(task_record, ni.DAQmx_Val_FiniteSamps,
                                  self.epoch_size)
        ni.DAQmxCfgDigEdgeStartTrig(task_record, trigger, ni.DAQmx_Val_Rising)

        # Ensure trigger rearms after each detection event
        ni.DAQmxSetStartTrigRetriggerable(task_record, True)
        duration = self.trigger_duration*0.5
        ni.DAQmxSetDigEdgeStartTrigDigFltrMinPulseWidth(task_record, duration)

        ni.DAQmxTaskControl(task_input, ni.DAQmx_Val_Task_Commit)
        ni.DAQmxTaskControl(task_record, ni.DAQmx_Val_Task_Commit)

        if callback is not None:
            self._everynsamples_cb = callback
            def event_cb(task, event_type, n_samples, data):
                self._everynsamples_cb()
                return 0

            samples = self.epoch_size
            log.debug('Configuring every N samples callback with %d samples',
                      samples)
            self._everynsamples_cb_ptr = \
                ni.DAQmxEveryNSamplesEventCallbackPtr(event_cb)
            event_type = ni.DAQmx_Val_Acquired_Into_Buffer
            ni.DAQmxRegisterEveryNSamplesEvent(task_input, event_type,
                                               samples, 0,
                                               self._everynsamples_cb_ptr, None)

        return task_input, task_record

    def read_timer(self):
        result = ctypes.c_uint32()
        ni.DAQmxReadCounterScalarU32(self._task_counter, 0, result, None)
        return result.value

    def read_analog(self, timeout=None, restart=False):
        if timeout is None:
            timeout = 0
        result = ctypes.c_uint32()
        ni.DAQmxGetReadAvailSampPerChan(self._task_analog, result)
        log.debug('%d s/chan available for %s', result.value, self.input_line)

        result = ctypes.c_long()
        samples = self.epoch_size*self.channels
        analog_data = np.empty(samples, dtype=np.double)
        ni.DAQmxReadAnalogF64(self._task_analog, self.epoch_size, timeout,
                              ni.DAQmx_Val_GroupByChannel, analog_data, samples,
                              result, None)
        log.debug('Read %d s/chan from %s', result.value, self.input_line)
        if restart:
            ni.DAQmxStopTask(self._task_analog)
            ni.DAQmxStartTask(self._task_analog)
        return analog_data.reshape((self.channels, -1))

    def samples_available(self):
        result = ctypes.c_uint32()
        ni.DAQmxGetReadAvailSampPerChan(self._task_analog, result)
        return result.value


class DAQmxSink(DAQmxBase, Sink):

    def __init__(self, fs=200e3, output_line='/Dev1/ao0',
                 trigger_line='/Dev1/port0/line0',
                 run_line='/Dev1/port0/line1', attenuator=None, **kwargs):
        variables = locals()
        kwargs = variables.pop('kwargs')
        for k, v in variables.items():
            setattr(self, k, v)
        DAQmxBase.__init__(self)
        Sink.__init__(self, **kwargs)
        self.write_size = int(self.fs * 1)

    def setup(self):
        self._task_analog, self._task_digital = \
            self._create_ao(self.output_line, self.trigger_line, self.run_line)
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

    def _create_ao(self, output_line, trigger_line, run_line):
        # Set up analog task
        task_analog = self._create_task('analog output')
        task_digital = self._create_task('digital output')

        # Setup analog output
        ni.DAQmxCreateAOVoltageChan(task_analog, output_line, '', -10, 10,
                                    ni.DAQmx_Val_Volts, '')
        ni.DAQmxSetWriteRegenMode(task_analog, ni.DAQmx_Val_DoNotAllowRegen)
        ni.DAQmxCfgSampClkTiming(task_analog, '', self.fs, ni.DAQmx_Val_Rising,
                                 ni.DAQmx_Val_ContSamps, int(self.fs))

        # Set up trigger line
        ni.DAQmxCreateDOChan(task_digital, ','.join([trigger_line, run_line]),
                             '', ni.DAQmx_Val_ChanPerLine)
        ni.DAQmxSetWriteRegenMode(task_digital, ni.DAQmx_Val_DoNotAllowRegen)
        ni.DAQmxCfgSampClkTiming(task_digital, 'ao/SampleClock', self.fs,
                                 ni.DAQmx_Val_Rising, ni.DAQmx_Val_ContSamps,
                                 int(self.fs))

        # Commit the tasks so we can catch resource errors early
        ni.DAQmxTaskControl(task_analog, ni.DAQmx_Val_Task_Commit)
        ni.DAQmxTaskControl(task_digital, ni.DAQmx_Val_Task_Commit)
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

    def write(self, attenuation, analog, trigger, running):
        analog_result = ctypes.c_long()
        digital_result = ctypes.c_long()
        ni.DAQmxWriteAnalogF64(self._task_analog, len(analog), False, -1,
                               ni.DAQmx_Val_GroupByChannel,
                               analog.astype(np.double), analog_result, None)

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

    def get_hw_sf(self, best_sf):
        if self.attenuator is None:
            raise ValueError, 'Cannot control attenuation'
        return self.attenuator.get_hw_sf(best_sf)


class DAQmxAttenControl(DAQmxBase):
    '''
    Handles configuration and control of niDAQmx for controlling volume via
    serial control of a programmable IC chip
    '''
    # Gain control settings in dB.  This is IC specific.
    VOLUME_STEP = 0.5
    VOLUME_MAX = 31.5
    VOLUME_MIN = -95.5
    VOLUME_BITS = 16

    # List of all possible gains that can be realized via hardware setting.
    HW_GAINS = np.arange(VOLUME_MIN, VOLUME_MAX+VOLUME_STEP, VOLUME_STEP)
    HW_SF = 10**(HW_GAINS/20.0)

    # This is limited by the input characteristics of the IC (right now the
    # chip needs at least 90us from the time the chip select line goes low to
    # reading the first sample from the serial input.  In addition, it needs a
    # 20us set-up time for reading the next sample (i.e. the clock line must
    # not go high until 20us after the sample line goes high, and then it must
    # hold the sample line high for at least 20us).  This places an upper bound
    # of 10e6 on the clock frequency.  However, a more reasonable upper bound
    # is probably 1e6 (1GHz).  Minimum frequency is 6250 (100 kHz timebase/16).
    VOLUME_CLOCK_FREQ = 1e5

    # Each bit is represented by two samples in the buffer to allow the serial
    # clock to transition while the serial data line is held constant at the
    # control value.  Two additional bits are added (one at the beginning and
    # end of the signal) to bring the chip select line back to high.
    # Samp     1   2   3
    # Value    1   0   1
    # P0.0  1 0 0 0 0 0 0 1 - IC reads serial data when 0
    # P0.1  0 0 1 0 1 0 1 1 - IC reads a bit from P0.2 when 0 -> 1
    # P0.2  0 1 1 0 0 1 1 0 - Hold value at 0 or 1 when P0.1 0 -> 1
    VOLUME_BUFFER = VOLUME_BITS * 2 + 2

    fs = 200e3

    def __init__(self, clock_line, cs_line, data_line, mute_line, zc_line,
                 hw_clock='/Dev1/FreqOut'):
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
        super(DAQmxAttenControl, self).__init__()

    def _setup_serial_timer(self, timer):
        # Configure timing of digital output using a counter that runs at twice
        # the frequency of the sample clock (enables us to set the value of the
        # serial data line high or low one cycle before setting the serial
        # clock high.  Here, the serial clock basically runs at half the rate
        # of the sample clock.
        task = self._create_task('COM timer')
        ni.DAQmxCreateCOPulseChanFreq(task, timer, '', ni.DAQmx_Val_Hz,
                                      ni.DAQmx_Val_Low, 0,
                                      self.VOLUME_CLOCK_FREQ, 0.5)
        ni.DAQmxCfgImplicitTiming(task, ni.DAQmx_Val_ContSamps,
                                  self.VOLUME_BUFFER)
        ni.DAQmxTaskControl(task, ni.DAQmx_Val_Task_Commit)
        return task

    def _setup_serial_comm(self, volume_lines, timer=None):
        task = self._create_task('Serial COM')
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
        self._task_mute = self._create_task('Mute control')
        self._task_zc = self._create_task('ZC control')

        self._task_com, self._task_clk = \
            self._setup_serial_comm(self.serial_line, self.hw_clock)

        # Set up mute line.  In theory we can combine both of these lines into
        # a single task, but it's easier to program them separately.
        ni.DAQmxCreateDOChan(self._task_mute, self.mute_line, 'mute',
                             ni.DAQmx_Val_ChanPerLine)
        ni.DAQmxWriteDigitalScalarU32(self._task_mute, True, -1, 1, None)
        ni.DAQmxWaitUntilTaskDone(self._task_mute, -1)

        # Set up zero crossing line
        ni.DAQmxCreateDOChan(self._task_zc, self.zc_line, 'zc',
                             ni.DAQmx_Val_ChanForAllLines)
        ni.DAQmxWriteDigitalScalarU32(self._task_zc, True, -1, 1, None)
        ni.DAQmxWaitUntilTaskDone(self._task_zc, -1)

        # Commit the tasks now (to catch errors early)
        ni.DAQmxTaskControl(self._task_mute, ni.DAQmx_Val_Task_Commit)
        ni.DAQmxTaskControl(self._task_zc, ni.DAQmx_Val_Task_Commit)

        if self._task_clk is not None:
            ni.DAQmxStartTask(self._task_clk)

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
            Desired gain in dB.  For attenuation, pass a negative value.

        Returns
        -------
        byte : int
            Byte value to send via serial interface.

        Raises
        ------
        ValueError if the requested gain is outside valid boundaries or is not
        a multiple of the gain step size.
        '''
        if gain > self.VOLUME_MAX:
            raise ValueError('Requested gain is too high')
        if gain < self.VOLUME_MIN:
            raise ValueError('Requested gain is too low')
        if gain % self.VOLUME_STEP != 0:
            raise ValueError('Requested gain is not multiple of step size')
        return int(255-((31.5-gain)/0.5))

    def _build_com_signal(self, word):
        bitword = [(word >> i) & 1 for i in range(self.VOLUME_BITS)]
        samples = self.VOLUME_BITS*2
        line_csi = np.zeros(samples, dtype=np.uint8)  # chip select line
        line_clk = np.zeros(samples, dtype=np.uint8)  # sample clock line
        line_sdi = np.zeros(samples, dtype=np.uint8)  # sample data line
        line_clk[1::2] = 1
        for i, bit in enumerate(bitword):
            offset = i*2
            line_sdi[offset:offset+2] = bit

        line_csi = np.pad(line_csi, 1, mode='constant', constant_values=1)
        line_clk = np.pad(line_clk, 1, mode='constant', constant_values=0)
        line_sdi = np.pad(line_sdi, 1, mode='constant', constant_values=0)
        return np.vstack((line_csi, line_clk, line_sdi))

    def set_gain(self, right, left):
        '''
        Set the IC volume control to the desired gain.

        This function will not return until the serial value has been written
        to the digital output.  There is no way to programatically check that
        the IC is configured properly.

        Parameters
        ----------
        right : float
            Desired gain for right channel in dB.  For attenuation, pass a
            negative value.
        left : float
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
        log.debug('Setting gain to %f (right) and %f (left)', right, left)
        rbyte = self._gain_to_byte(right)
        lbyte = self._gain_to_byte(left)
        word = rbyte | (lbyte << 8)
        com_signal = self._build_com_signal(word)
        result = ctypes.c_int32()
        ni.DAQmxWriteDigitalLines(self._task_com, self.VOLUME_BUFFER, False,
                                  -1, ni.DAQmx_Val_GroupByChannel,
                                  com_signal.ravel().astype(np.uint8), result,
                                  None)
        ni.DAQmxStartTask(self._task_com)
        ni.DAQmxWaitUntilTaskDone(self._task_com, 1)
        ni.DAQmxStopTask(self._task_com)

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
        ni.DAQmxWriteDigitalScalarU32(self._task_mute, True, -1, int(mute),
                                      None)
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
        ni.DAQmxWriteDigitalScalarU32(self._task_zc, True, -1,
                                      int(zero_crossing), None)
        ni.DAQmxWaitUntilTaskDone(self._task_zc, 1)

    def get_hw_sf(self, sf):
        '''
        Return closest attenuation (as a scaling factor) that can be realized
        via hardware.
        '''
        mask = self.HW_SF < sf
        if mask.any():
            return self.HW_SF[mask][-1]
        return self.HW_SF[0]
