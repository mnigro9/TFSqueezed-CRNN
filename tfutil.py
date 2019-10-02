import os
import glob
import numpy as np
import random
import soundfile as sf  #reading 24 bit sound file
import librosa          # stft function


def createPSD(sound, fs):
    '''
    The window length is designed for high sampling frequency audio recordings such as 44.1kHz (Voice), 48kHz (DCASE) and 50 kHz (Saarbrucket)
    Input: 
        sound: audio signal 
        fs: sample frequency
    return:
        tDim:   time-slots, time vector of the psd matrix
        fDim:   frequency bins, frequency vector of the psd matrix
        psd:    power spectrum density (PSD) matrix
        maxpsd: maximum value of the psd matrix
   ''' 


    # Define STFT parameters
    framesize = 0.02 #Setting frame size to 2 msec
    n_fft=max(np.floor(framesize*fs),1024)
    overlap = 2
    hop_length = int(n_fft/overlap)
    win_length = n_fft
    center = True
    stft_config = dict(n_fft=n_fft, 
                    hop_length=hop_length, 
                    win_length=n_fft, 
                    window='hann')
    lenSound = int(np.floor(len(sound)/fs)*fs)
    fDim = int(np.floor(n_fft/2) + 1)
    tDim = int(lenSound/(fDim-1))+ 1
    
    # calculate psd
    STFT = librosa.stft(sound, **stft_config)  
    
    mag = np.asarray(np.abs(STFT))
    tmp = np.multiply(mag,mag)
    maxpsd = np.max(tmp)
    psd = np.asarray(tmp/maxpsd*255)

    return tDim, fDim, psd, maxpsd



def truncatePSD(psdMatrix, threshold=2):
    # Truncating all values above the threshold and normalize it over the [0:threshold]

    maxPixelValue = 255   #uint8 
    
    truncNormPSD = psdMatrix;
    truncNormPSD[psdMatrix>threshold] = threshold
    truncNormPSD = truncNormPSD/threshold*maxPixelValue
    
    return truncNormPSD



def groupTimePSD(psdMatrix, N=5):
    # Grouping N time-vectors together into one timeslot

    nrow, ncol = psdMatrix.shape
    numslots = int(np.floor(ncol/N))
    timeFramedPSD = np.zeros([nrow,numslots])

    for i in range(numslots):
        tmp = psdMatrix[0:nrow,i*N:i*N+N]
        timeFramedPSD[:,i]=np.sum(tmp,axis=1)

    return timeFramedPSD



def groupFreqPSD(psdMatrix, Nupper=5, Nlower=2, lowerbins=50):
    '''
    Grouping N Frequency-vectors together into one frequency bin
    NOTE: Equal bin distance does not work well for low frequencies. 
    New changes: only grouping bins above index 50 for every 5 bins 
                 and 2 bins for lower index
    '''
    
    nrow, ncol = psdMatrix.shape
    numupperbins = int(np.floor((nrow-lowerbins)/Nupper))
    numlowerbins = int(np.floor(lowerbins/2))
    freqFramedPSD = np.zeros([numupperbins+numlowerbins,ncol])

    for i in range(numlowerbins):
        freqFramedPSD[i] = psdMatrix[Nlower*i] +psdMatrix[Nlower*i+1]
    
    for i in range(numupperbins):
        startbin=i*Nupper+numlowerbins
        stopbin = (i+1)*Nupper+numlowerbins
        tmp = psdMatrix[startbin:stopbin]
        freqFramedPSD[i+numlowerbins,:]=np.sum(tmp,axis=0)
    
    return freqFramedPSD

