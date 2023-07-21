import sys
import numpy as np
import scipy as sp
#from Utils.plots import visualize_segmentation

class IMS:

    """
    IMS: class that implements the IMS algorithm, a powerful tool
    for segmenting PPG data into pulses, helping extracting RIV information.

    compute_IMS: implements the proper IMS part of the segmentation algorithm.

    compute_artifacts: determines wjich samples, in the RIV signals, are to be
    interpreted as respiratory artifacts.
    """

    def __init__(self, m, Fs, rFs, dur, N, par, visualize):
        self.m = m                                  #IMS evalates every m-distant-sample intensities in the compute RR imput signal
        self.Fs = Fs                                #sample rate of the compute_IMS input signal
        self.rFs = rFs                              #sample rate of the RIV signals
        self.dur = dur                              #duration (sec) of the compute_IMS input signal
        self.N = N                                  #number of samples of the compute_IMS input signal
        self.a_sl_lw_par = par[0]                   #lw adaptation parameter slow - low (0.6)
        self.a_fs_lw_par = par[1]                   #lw adaptation parameter fast - low (0.5)
        self.a_sl_up_par = par[2]                   #up adaptation parameter slow - up (2.0)
        self.a_fs_up_par = par[3]                   #up adaptation parameter fast - up (1.6)
        self.riv_samples = round(rFs*dur)
        self.artifacts = np.zeros(self.riv_samples)          #data structure to store which RIV samples are referred to respiratory artifacts
        self.peak_interval_res = np.zeros(self.riv_samples)  #data structure which stores the RIFV signal
        self.peak_val_max_res = np.zeros(self.riv_samples)   #data structure which stores the RIIV signal
        self.amplitude_max_res = np.zeros(self.riv_samples)  #data structure which stores the RIAV signal
        self.visualize = visualize                  #set to TRUE to visualize intermediate plots (not recommended)


    def compute_artifacts(self):
        lamb = 0                                    #counter of the number of subsequent artifacts
        th_lw = 0.6*self.amplitude_max_res[0]       #lower threshold
        th_up = 1.4*self.amplitude_max_res[0]       #upper threshold
        for i in range(1,self.riv_samples):        #for each sample, if the RIV signals do not satify certain conditions...
            if (self.peak_interval_res[i]<0.23 or self.peak_interval_res[i]>=2.4) or (self.peak_interval_res[i]<=self.peak_interval_res[i-1]/2) or (self.amplitude_max_res[i]<th_lw or self.amplitude_max_res[i]>th_up):
                self.artifacts[i] = 1               #detect the artifact
                if lamb>0:                          #if the counter triggers the updating procedure
                    th_lw = (th_lw + self.amplitude_max_res[i]*self.a_sl_lw_par)/2  #update the lower threshold with the a_sl_lw_par parameter
                    th_up = self.amplitude_max_res[i]*self.a_sl_up_par              #update the upper threshold with the a_sl_up_par parameter
                lamb = lamb+1                       #update the counter
            else:                                   #otherwise...
                th_lw = (th_lw + self.amplitude_max_res[i]*self.a_fs_lw_par)/2      #update the lower threshold with the a_fs_lw_par parameter
                th_up = self.amplitude_max_res[i]*self.a_fs_up_par                  #update the lower threshold with the a_fs_up_par parameter
                lamb = 0


    def compute_IMS(self, x):
        out = np.empty(0)                    #array to store the final segmented signal
        seg = 0                              #segments counter, it is independent of the specific z-th line to which the current segment is assigned
        z = 0                                #lines counter
        seg_in_line = 0                      #number of segments assigned to the z-th line

        line_prev = np.linspace(x[seg], x[seg+self.m-1], self.m)  #starting from x[0], evaluate the next m-distant-sample intesity x[self.m-1] and build a line that connects those two points
        dx_prev = self.m-1                                        #distance between the projections of x[0] and x[self.m-1] onto the x-axis
        dy_prev = x[seg+self.m-1]-x[seg]                          #distance between the projections of x[0] and x[self.m-1] onto the y-axis
        a_prev = dy_prev/dx_prev                                  #slope of the [x[0],x[self.m-1]] segment

        z = z+1                                                   #increment lines counter
        seg = seg+1                                               #increment segments counter
        seg_in_line = seg_in_line+1                               #increment number-of-segments-in-current-line counter

        while (seg+1)*self.m<self.N:                                                                                #until the end of the x signal
            line_next = np.linspace(x[seg*self.m], x[(seg+1)*self.m-1], self.m)                                     #starting from x[seg], evaluate the next m-distant-sample intesity x[(seg+1)*self.m-1] and build a line that connects those two points
            dx_next = self.m-1                                                                                      #distance between the projections of x[seg] and x[(seg+1)*self.m-1] onto the x-axis
            dy_next =  x[(seg+1)*self.m-1]-x[seg*self.m]                                                            #distance between the projections of x[seg] and x[(seg+1)*self.m-1] onto the y-axis
            a_next = dy_next/dx_next                                                                                #slope of the [x[0],x[self.m-1]] segment
            if (a_prev*a_next>0) or (a_prev==0 and a_next==0):                                                      #if the previous-segment slope and the surrent-segment slope have the same sign...
                line_prev = np.linspace(x[(seg-seg_in_line)*self.m], x[(seg+1)*self.m-1], (seg_in_line+1)*self.m)       #consider the two segments part of the same line, i.e. starting from starting from x[(seg-seg_in_line)*self.m] evaluate the next
                                                                                                                        #(seg_in_line+1)*m-distant-sample intesity x[(seg+1)*self.m-1] and build a line that connects those two points
                dx_prev = (seg_in_line+1)*self.m-1                                                                      #distance between the projections of x[(seg-seg_in_line)*self.m] and x[(seg+1)*self.m-1] onto the x-axis
                dy_prev = x[(seg+1)*self.m-1]-x[(seg-seg_in_line)*self.m]                                               #distance between the projections of x[(seg-seg_in_line)*self.m] and x[(seg+1)*self.m-1] onto the y-axis
                a_prev = dy_prev/dx_prev                                                                                #slope of the [x[(seg-seg_in_line)*self.m],x[(seg+1)*self.m-1]] segment
                seg = seg+1                                                                                             #increment the segment counter (not the line counter, since the new segment still belongs to the current line)
                seg_in_line = seg_in_line+1                                                                             #increment the number-of-segments-in-current-line counter
            else:                                                                                                   #if the previous-segment slope and the surrent-segment slope have different sign...
                z=z+1                                                                                                   #consider the two segments part of different lines, i.e. increment the lines counter and
                out = np.concatenate((out, line_prev))                                                                  #add the previous line to the result array
                line_prev = line_next                                                                                   #set the current line to the new 'previous' line
                dy_prev = dy_next                                                                                       #distance between the projections of the current line (new previous line) onto the x-axis
                dx_prev = dx_next                                                                                       #distance between the projections of the current line (new previous line) onto the y-axis
                a_prev = a_next                                                                                         #slope of the current line (new previous line)
                seg = seg+1                                                                                             #increment the segment counter
                seg_in_line = 1                                                                                         #increment the number-of-segments-in-current-line counter

        z=z+1                                                                                                           #increment the line-counter for the last time
        out = np.concatenate((out, line_prev))                                                                          #add the last line to the result array
        seg = seg+1                                                                                                     #increment the segment counter for the last time
        seg_in_line = 1                                                                                                 #increment the number-of-segments-in-current-line counter for the last time

        #local maxima and minima detection phase, performed on the resulting
        #IMS segmented signal, to further simplify the waveform and to make
        #the RIV computation easier.

        out_x = np.linspace(0, out.size, out.size)
        peaks_idx_max, prop_max = sp.signal.find_peaks(out)
        peaks_idx_min, prop_min = sp.signal.find_peaks(-out)
        if peaks_idx_min.size > peaks_idx_max.size:
            peaks_idx_max = np.concatenate([peaks_idx_max, [out.size-1]])
        if peaks_idx_min.size < peaks_idx_max.size:
            peaks_idx_min = np.concatenate([[0], peaks_idx_min])

        peaks_idx = np.concatenate([peaks_idx_max, peaks_idx_min])
        peaks_idx.sort(kind='mergesort')

        #if self.visualize:
        #    visualize_segmentation(x, out, out_x, peaks_idx, self.N)

        #RIFV is computed as an evenly 4Hz-sampled and
        #minimum-peak-time-interspersed series

        peak_interval = np.diff(peaks_idx_min/self.Fs)
        self.peak_interval_res = sp.signal.resample(peak_interval, self.riv_samples)

        #RIIV information is conveyed by the maximum-peak-valued
        #and 4Hz-sampled time series

        peak_val_max = out[peaks_idx_max]
        self.peak_val_max_res = sp.signal.resample(peak_val_max, self.riv_samples)

        #RIAV is carried by 4Hz-sampled series generated from the
        #difference between maximum values and minimum values (amplitude trend)

        amplitude_max = out[peaks_idx_max] - out[peaks_idx_min]
        self.amplitude_max_res = sp.signal.resample(amplitude_max, self.riv_samples)

        #artifacts computation
        self.compute_artifacts()


    def get_IMS(self):
        return self.peak_interval_res, self.peak_val_max_res, self.amplitude_max_res, self.artifacts