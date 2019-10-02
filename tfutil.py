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



def truncatePSD(psdMatrix, threshold2=5):
  # Scaling values in [0,1] to [0-220], (1,5] to [220,255],
  # and values greater than 5 will be assigned to 255
  # The result is a truncated-scaled matrix of the original matrix

  threshold1 = 1
  lowPixelValue = 220

  #threshold2 = 5
  highPixelValue = 255   #uint8 

  
  psdMatrix = np.asarray(psdMatrix)
  nrow,ncol = psdMatrix.shape
  truncNormPSD = np.zeros((nrow, ncol))
  # Checking if the value is below thresold1 and perform scaling
  # Masking is used to identify the elements that satisfies the criteria
  mask1 = np.zeros((nrow, ncol))
  mask1[psdMatrix<threshold1] = 1
  tmp1 = psdMatrix/threshold1*lowPixelValue
  # mask1 indicates elements that were originally less than threshold1
  tmp1 = np.multiply(tmp1,mask1) 
  truncNormPSD = tmp1

  # Checking if the values are above threshold 2
  mask2 = np.zeros((nrow,ncol))
  truncNormPSD[psdMatrix>threshold2] = 255
  mask2[psdMatrix>threshold2] = 1

  # The rest of the numbers that between threshold1 and threshold2 will be scaled
  mask3 = np.ones((nrow,ncol))
  mask3 = mask3-mask1-mask2 #The rest of the elements that fall between threshold1-threshold2
  tmp2 = np.zeros((nrow,ncol))
  tmp2 = (psdMatrix-threshold1)/(threshold2-threshold1)*(highPixelValue-lowPixelValue)+lowPixelValue
  tmp2 = np.multiply(tmp2,mask3)

  truncNormPSD=truncNormPSD+tmp2
  #truncNormPSD = truncNormPSD/threshold*maxPixelValue
  
  return truncNormPSD



def groupTimePSD(psdMatrix, N=5):
    # Squeezing N time-vectors together into one timeslot
    # The result values are the sum of the N vectors
    # The returning matrix time-column is reduced by N

    nrow, ncol = psdMatrix.shape
    numslots = int(np.floor(ncol/N))
    timeFramedPSD = np.zeros([nrow,numslots])

    for i in range(numslots):
        tmp = psdMatrix[0:nrow,i*N:i*N+N]
        timeFramedPSD[:,i]=np.sum(tmp,axis=1)

    return timeFramedPSD




def groupFreqPSD(psdMatrix, Nupper=5):
  # Squeezing the frequency bins into one according to the frequency band
  # Frequency bins in the lowerbin will be squeezed from 2 into 1
  # Frequency bins in the upperbins will be squeezed from Nupper into 1
  # The returning matrix freq-row is reduced by about 4
  
  lowerbins = 50;
  Nlower = 2

  nrow, ncol = psdMatrix.shape
  
  # Define the size of new dimension with the frequency squeezing
  numupperbins = int(np.floor((nrow-lowerbins)/Nupper))
  numlowerbins = int(np.floor(lowerbins/2))
  freqFramedPSD = np.zeros([numupperbins+numlowerbins,ncol])

  # Squeezing the lower frequency bins
  for i in range(numlowerbins):
    freqFramedPSD[i] = psdMatrix[Nlower*i] +psdMatrix[Nlower*i+1]

  # Squeezing the upper frequency bins
  for i in range(numupperbins):
    startbin=i*Nupper+numlowerbins
    stopbin = (i+1)*Nupper+numlowerbins
    tmp = psdMatrix[startbin:stopbin]
    freqFramedPSD[i+numlowerbins,:]=np.sum(tmp,axis=0)
  return freqFramedPSD
