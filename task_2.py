import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np


class AudioWorker:

    __filename = None
    __freq_const = None

    __audio = None
    __audio_transformed = None
    __sr = None
    __freq = None
    __freq_transformed = None
    __spec = None
    __spec_transformed = None

    def __init__(self, filename, freq):
        self.__filename = filename
        self.__freq_const = freq

    def __read_file(self):
        self.__audio, self.__sr = librosa.load(self.__filename, sr=None)

    def __write_audio(self):
        sf.write('transformed.wav', self.__audio_transformed, self.__sr)

    def __calculate_freq(self):
        self.__spec = np.fft.fft(self.__audio)
        self.__freq = np.fft.fftfreq(len(self.__spec)) * self.__sr

    def __calculate_transformed_freq(self):
        self.__spec_transformed = np.fft.fft(self.__audio_transformed)
        self.__freq_transformed = np.fft.fftfreq(len(self.__spec_transformed)) * self.__sr

    def __run_transformation(self):
        tmp = self.__spec.copy()
        tmp[np.abs(self.__freq) < self.__freq_const] = 0
        self.__audio_transformed = np.real(np.fft.ifft(tmp))

    def __draw(self):
        plt.figure(figsize=(10, 6))

        plt.plot(np.arange(len(self.__audio)) / self.__sr,
                 self.__audio, label='Исходный')
        plt.plot(np.arange(len(self.__audio_transformed)) / self.__sr,
                 self.__audio_transformed, label='Фильтрованный')

        plt.title('Графики исходного и фильтрованного сигналов')
        plt.xlabel('Время')
        plt.ylabel('Амплитуда')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def __draw_freq(self):
        plt.figure(figsize=(10, 6))

        plt.plot(self.__freq, np.abs(self.__spec), label='Исходный')
        plt.plot(self.__freq_transformed, np.abs(self.__spec_transformed), label='Фильтрованный')
        plt.xlim(-800, 800)
        plt.ylim(0, 3500)
        plt.title('Графики модулей образов')
        plt.xlabel('Частота')
        plt.ylabel('Значение')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        self.__read_file()
        self.__calculate_freq()
        self.__run_transformation()
        self.__calculate_transformed_freq()
        self.__draw()
        self.__draw_freq()
        self.__write_audio()


if __name__ == '__main__':
    audio_worker = AudioWorker('MUHA.wav', 321)
    audio_worker.run()
