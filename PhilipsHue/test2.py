import numpy as np
import pyaudio
import time
import librosa
import ipdb

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        np_buffer = np.frombuffer(in_data, dtype=np.float32)        
        #print(librosa.feature.mfcc(np_buffer))
        pitch_block = librosa.core.pitch.estimate_tuning(np_buffer, sr=self.RATE)
        pitch_block = np.nan_to_num(pitch_block)
        hz = max(np.average(pitch_block), 1)
        # import ipdb;ipdb.set_trace()

        # m_block = librosa.feature.melspectrogram(np_buffer, sr=self.RATE,
        #                                      n_fft=2048,
        #                                      hop_length=2048,
        #                                      center=False)         
        # avg_m_block = np.average(m_block)        
        # hz = librosa.mel_to_hz(avg_m_block)        
        
        note = librosa.core.hz_to_note(hz)
        print(note, hz)
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(2)


audio = AudioHandler()
audio.start()     # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()


# y, sr = librosa.load(librosa.util.example_audio_file())
# m_block = librosa.feature.melspectrogram(y, sr=sr,
#                                              n_fft=2048,
#                                              hop_length=2048,
#                                              center=False) 
# ipdb.set_trace()
# pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
# print(pitches, magnitudes)
# S = np.abs(librosa.stft(y))
# print(librosa.estimate_tuning(S=S, sr=sr))

