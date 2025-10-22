""" Partitioned-Block-Based Frequency Domain Kalman Filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft
import sys

class PFDKF:
    def __init__(self, N, M, A=0.999, P_initial=10):
        self.N = N
        self.M = M
        self.A = A
        self.A2 = A**2
        self.m_soomth_factor = 0.5

        self.x = np.zeros(shape=(2*self.M), dtype=np.float32)

        self.m = np.zeros(shape=(self.M + 1), dtype=np.float32)
        self.P = np.full((self.N, self.M + 1), P_initial)
        self.X = np.zeros((self.N, self.M + 1), dtype=complex)
        self.H = np.zeros((self.N, self.M + 1), dtype=complex)
        self.mu = np.zeros((self.N, self.M + 1), dtype=complex)
        self.half_window = np.concatenate(([1]*self.M, [0]*self.M))


    def filt(self, x, d):
        assert(len(x) == self.M)
        self.x = np.concatenate([self.x[self.M:], x])
        X = fft(self.x)
        self.X[1:] = self.X[:-1]
        self.X[0] = X
        Y = np.sum(self.H*self.X, axis=0)
        y = ifft(Y).real[self.M:]
        e = d-y

        e_fft = np.concatenate((np.zeros(shape=(self.M,), dtype=np.float32), e))
        self.E = fft(e_fft)
        X2 = np.sum(np.abs(self.X)**2, axis=0)
        self.m = self.m_soomth_factor * self.m + (1-self.m_soomth_factor) * np.abs(self.E)**2
        R = np.sum(self.X*self.P*self.X.conj(), 0) + 2*self.m/self.N
        self.mu = self.P / (R + 1e-10)
        W = 1 - np.sum(self.mu*np.abs(self.X)**2, 0)
        E_res = W*self.E
        e = ifft(E_res).real[self.M:].real
        y = d-e
        return e, y

    def update(self):
        G = self.mu*self.X.conj()
        self.P = self.A2*(1 - 0.5*G*self.X)*self.P + (1-self.A2)*np.abs(self.H)**2
        self.H = self.A*(self.H + fft(self.half_window*(ifft(self.E*G).real)))


def pfdkf(x, d, N=10, M=256, A=0.999, P_initial=10):
    ft = PFDKF(N, M, A, P_initial)
    num_block = min(len(x), len(d)) // M

    e = np.zeros(num_block*M)
    y = np.zeros(num_block*M)
    for n in range(num_block):
        x_n = x[n*M:(n+1)*M]
        d_n = d[n*M:(n+1)*M]
        e_n, y_n = ft.filt(x_n, d_n)
        ft.update()
        e[n*M:(n+1)*M] = e_n
        y[n*M:(n+1)*M] = y_n
    return e, y

def process(ref_path, mic_path, sr=16000):
    import soundfile as sf
    mic ,sr= sf.read(mic_path)
    ref ,sr= sf.read(ref_path)
    error, echo = pfdkf(ref, mic, N=10, M=400, A=0.999, P_initial=10)
    sf.write('./mic.wav', mic, sr)
    sf.write('./ref.wav', ref, sr)    
    sf.write('./e.wav', error, sr)
    sf.write('./y.wav', echo, sr)
    print(f'mic: {mic.shape}')
    print(f'ref: {ref.shape}')
    print(f'e: {error.shape}')
    print(f'y: {echo.shape}')


if __name__ == "__main__":
    mic = '/home/node25_tmpdata/xcli/percepnet/train/linear_model/mic.wav'
    ref = '/home/node25_tmpdata/xcli/percepnet/train/linear_model/ref.wav'
    process(ref, mic)