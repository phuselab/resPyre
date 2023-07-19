import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class FIRfilt :

		"""

		FIRfilt: class that implements a L-tap FIR type I high-pass or low-pass filter
		with a 74dB stop band attenuation (Blckman window). The filter is built
		using the "window" method.

		filtering: filters the x input signal with an instance of the L-tap FIR type I
		high-pass or low-pass filter.

		"""

		def __init__(self, type, fc, ft, Fs, N):
			self.type = type																									#type of the filter "high" or "low"
			self.wt = (ft*2*np.pi)/Fs																				 #transition bandwidth (rad)
			self.wc = (fc*2*np.pi)/Fs																				 #cutoff frequency (rad)

			self.L = int(np.ceil(11*np.pi/self.wt))													 #transition band support (in number of samples) according to the
																																				#choice of the Blackman window (i.e. the desired stop band attenuation).
																																				#L-tap: L determines also the width of the truncated impulse response

			self.L = self.L+1 if self.L%2==0 else self.L											#since L must be odd for FIR Type I filters, if L is even, set L = L+1
			self.M = self.L-1																								 #M is interchanging factor between L and alpha
			self.alpha = self.M//2																						#alpha is the integer (since L is odd) ammount of delayed/traslated samples
																																				#that allows the impulse response to be symmetric (i.e. allows the filter
																																				#to produce a linear phase filtering)

			n = np.linspace(0, self.L, self.L)																#L evenly spaced samples (impulse response support)

			h_id = self.wc/np.pi*np.sinc(self.wc/np.pi*(n-self.alpha))				#low-pass ideal impulse response
			h_l = np.blackman(self.L)																				 #Blackman window

			if self.type=='HP':																							 #The high-pass impulse response is obtained simply flipping the low-pass
				h_id = -h_id																										#impulse response and setting the central-sample amplitude to 1-2*fc/Fs
				h_id[self.alpha+1] = 1-2*fc/Fs
			elif self.type=='LP':
				h_id[self.alpha+1] = 2*fc/Fs

			h_pre = h_id*h_l																									#window truncation of the ideal impulse response
			self.h = np.concatenate([h_pre, np.zeros(np.abs(N-h_pre.size))])	#complete the impulse response to N samples (zero padding)
							

		def filtering(self, x):
			x_pad = np.concatenate([x[::-1], x, x[::-1]])																		 #pad the input signal x appending and prefixing a reversed version of x
			paddly = x_pad[x_pad.size-self.alpha:]																						#select the last alpha samples of the padded x

			x_pad_filt = sp.signal.lfilter(self.h, 1, np.concatenate([x_pad, paddly[::-1]]))	#concatenate the padded x signal with the reversed verion of the last alpha
																																												#samples of the same signal, and filter it with the previously built impulse response
			x_depaddly = x_pad_filt[self.alpha:]																							#traslate back the alpha-samples-shifted filtered signal
			x_filt = x_depaddly[x.size:x_depaddly.size-x.size]																#remove the pad, which purpose was the avoidace of edge effects that may occure during the filtering
			return x_filt
