import numpy as np
import matplotlib.pyplot as plt


class HardFilters:

    __a = 15
    __t1 = -6
    __t2 = 6
    __T = 36
    __dt = 0.01

    __b = None
    __c = None
    __d = None
    __temp = None
    __v_list = []

    __time_list = []
    __g_func = None
    __u_func = None
    __u_transformed_func = None
    __filtered_func = None
    __filtered_transformed_func = None
    __frequencies = None
    __task_type = 1

    def __init__(self, b, c, d, v_list, temp, task_type):
        self.__b = b
        self.__c = c
        self.__d = d
        self.__v_list = v_list
        self.__temp = temp
        self.__task_type = task_type

    def __calculate_time(self):
        self.__time_list = np.arange(
            -self.__T/2,
            self.__T/2,
            self.__dt
        )

    def __calculate_g(self):
        self.__g = np.zeros_like(self.__time_list)
        self.__g[(self.__time_list >= self.__t1) & (self.__time_list <= self.__t2)] = self.__a

    def __calculate_u(self):
        self.__u_func = (self.__g +
                         self.__b * (np.random.rand(len(self.__time_list)) - 0.5) +
                         self.__c * np.sin(self.__d * self.__time_list))

    def __calculate_fourier(self):
        self.__u_transformed_func = np.fft.fft(self.__u_func)
        self.__frequencies = np.fft.fftfreq(len(self.__time_list), self.__dt)

    def __remove_high_freq(self):
        self.__filtered_transformed_func = self.__u_transformed_func.copy()
        self.__filtered_transformed_func[np.abs(self.__frequencies) > self.__v_list[0]] = 0
        self.__filtered_func = np.fft.ifft(self.__filtered_transformed_func).real

    def __add_mask(self):
        self.__filtered_transformed_func = self.__u_transformed_func.copy()
        self.__filtered_transformed_func[
            np.logical_or(
                np.logical_and(
                    np.abs(self.__frequencies) >= self.__v_list[1],
                    np.abs(self.__frequencies) <= self.__v_list[2]
                ),
                np.abs(self.__frequencies) >= self.__v_list[3]
            )
        ] = 0
        self.__filtered_func = np.fft.ifft(self.__filtered_transformed_func).real

    def __add_next_mask(self):
        self.__filtered_transformed_func = self.__u_transformed_func.copy()
        self.__filtered_transformed_func[
            np.abs(self.__frequencies) <= self.__temp
        ] = 0
        self.__filtered_func = np.fft.ifft(self.__filtered_transformed_func).real

    def __draw(self):
        plt.figure(figsize=(10, 6))
        plt.title('Сравнение сигналов')

        plt.plot(self.__time_list, self.__u_func, label='Исходный')
        plt.plot(self.__time_list, self.__filtered_func, label='Фильтрованный')

        plt.legend()
        plt.show()

    def __draw_abs(self):
        plt.figure(figsize=(10, 6))
        plt.title('Сравнение модулей сигналов')

        plt.plot(self.__frequencies, np.abs(self.__u_transformed_func), label='Исходный')
        plt.plot(self.__frequencies, np.abs(self.__filtered_transformed_func), label='Фильтрованный')

        plt.ylim(0, 3000)

        if self.__task_type == 1:
            plt.xlim(-self.__v_list[0] - 3, self.__v_list[0] + 3)
        elif self.__task_type == 2:
            plt.xlim(-self.__v_list[3] - 2, self.__v_list[3] + 2)

        plt.legend()
        plt.show()

    def run(self):
        self.__calculate_time()
        self.__calculate_g()
        self.__calculate_u()
        self.__calculate_fourier()
        if self.__task_type == 1:
            self.__remove_high_freq()
        elif self.__task_type == 2:
            self.__add_mask()
        else:
            self.__add_next_mask()
        self.__draw()
        self.__draw_abs()


if __name__ == '__main__':

    # Task 1
    # hard_filters = HardFilters(4, 0, 5, [1], None, 1)

    # Task 2
    # hard_filters = HardFilters(3, 5, 9, [0], None, 2)

    # Task 3
    # hard_filters = HardFilters(4, 5, 9, [3], 2, 3)

    hard_filters.run()
