import pyaudio
import math
import struct
import time
import datetime
import matplotlib.pyplot as plt

TRIGGER_RMS = 100  # Adjusted threshold for graph
RATE = 16000  # sample rate
TIMEOUT_SECS = 3  # silence time after which recording stops
FRAME_SECS = 0.25  # length of frame(chunks) to be processed at once in secs
CUSHION_SECS = 1  # amount of recording before and after sound

SHORT_NORMALIZE = (1.0 / 32768.0)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SHORT_WIDTH = 2
CHUNK = int(RATE * FRAME_SECS)
CUSHION_FRAMES = int(CUSHION_SECS / FRAME_SECS)
TIMEOUT_FRAMES = int(TIMEOUT_SECS / FRAME_SECS)

class Recorder:
    @staticmethod
    def rms(frame):
        count = len(frame) / SHORT_WIDTH
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=CHUNK)
        self.time = time.time()
        self.quiet = []
        self.quiet_idx = -1
        self.timeout = 0

        # Initialize matplotlib for live plotting
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x_data = []
        self.y_data = []
        self.line, = self.ax.plot(self.x_data, self.y_data, label='RMS Value')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('RMS Value')
        self.ax.set_title('Real-Time Audio RMS Level')
        self.ax.axhline(y=TRIGGER_RMS, color='r', linestyle='--')  # Add horizontal line for threshold
        self.start_time = time.time()
        self.alert_triggered = False  # To avoid repeated alerts

    def record(self):
        print('')
        sound = []
        start = time.time()
        begin_time = None
        while True:
            data = self.stream.read(CHUNK)
            rms_val = self.rms(data)
            if self.inSound(data):
                sound.append(data)
                if begin_time is None:
                    begin_time = datetime.datetime.now()
            else:
                if len(sound) > 0:
                    duration = math.floor((datetime.datetime.now() - begin_time).total_seconds())
                    self.write(sound, begin_time, duration)
                    sound.clear()
                    begin_time = None
                else:
                    self.queueQuiet(data)

            # Update the plot with new RMS value
            self.update_plot(rms_val)

    def queueQuiet(self, data):
        self.quiet_idx += 1
        if self.quiet_idx == CUSHION_FRAMES:
            self.quiet_idx = 0
        if len(self.quiet) < CUSHION_FRAMES:
            self.quiet.append(data)
        else:
            self.quiet[self.quiet_idx] = data

    def dequeueQuiet(self, sound):
        if len(self.quiet) == 0:
            return sound

        ret = []
        if len(self.quiet) < CUSHION_FRAMES:
            ret.append(self.quiet)
            ret.extend(sound)
        else:
            ret.extend(self.quiet[self.quiet_idx + 1:])
            ret.extend(self.quiet[:self.quiet_idx + 1])
            ret.extend(sound)

        return ret

    def inSound(self, data):
        rms = self.rms(data)
        curr = time.time()

        if rms > TRIGGER_RMS:
            self.timeout = curr + TIMEOUT_SECS
            return True

        if curr < self.timeout:
            return True

        self.timeout = 0
        return False

    def write(self, sound, begin_time, duration):
        sound = self.dequeueQuiet(sound)
        keep_frames = len(sound) - TIMEOUT_FRAMES + CUSHION_FRAMES
        recording = b''.join(sound[0:keep_frames])

        # Print detection details
        print(f'[+] Detected noise at {begin_time.strftime("%m/%d/%Y, %H:%M:%S")}')
        print(f'    Duration: {duration} seconds')
        print(f'    RMS Level: {self.rms(recording)}')

    def update_plot(self, rms_val):
        elapsed_time = time.time() - self.start_time
        self.x_data.append(elapsed_time)
        self.y_data.append(rms_val)

        # Update line color based on threshold
        color = 'red' if rms_val > TRIGGER_RMS else 'blue'
        self.line.set_color(color)

        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.axhline(y=TRIGGER_RMS, color='r', linestyle='--')  # Add horizontal line for threshold
        plt.draw()
        plt.pause(0.01)

        # Trigger alert if RMS value exceeds threshold
        if rms_val > TRIGGER_RMS and not self.alert_triggered:
            print("\n[ALERT] RMS value exceeded the threshold!")
            self.alert_triggered = True
        elif rms_val <= TRIGGER_RMS:
            self.alert_triggered = False

a = Recorder()
a.record()
