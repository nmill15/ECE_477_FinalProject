import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import scipy
import sounddevice as sd
from playsound import playsound
import time
from NMF_Function import myNMF




# Training Speech dictionary
x, sr = librosa.load("sounds.wav", sr=None) # Loading in audio file
print("Training Dictionary")

# Calculating the spectrogram of the audio file
S = librosa.stft(x,hop_length=256,win_length=1024, window='hamming', n_fft=1024)
S_mag, S_phase = librosa.magphase(S)
S_db = librosa.amplitude_to_db(S_mag)

# NMF Parameters
r = 3
nIter = 75

[W_Speech,H,KL] = myNMF(S_mag,r,nIter)

Y_Speech = np.dot(W_Speech,H) # Reconstructed Spectrogram

print(" Dictionary Trained ")



# Loading in Test Audio File
x, sr = librosa.load("beatboxtest.wav", sr=None) # Loading in audio file

# Calculating the spectrogram of the audio file
S = librosa.stft(x,hop_length=256,win_length=1024, window='hamming', n_fft=1024)
S_mag, S_phase = librosa.magphase(S)
S_db = librosa.amplitude_to_db(S_mag)

# NMF Parameters
r = 3
nIter = 75

[W_Combo,H,KL] = myNMF(S_mag,r,nIter,bUpdateW=0,initW=W_Speech)
print("NMF Completed")


numframes = np.floor(x.size/256).astype(int)



localmax = np.zeros((3,100), dtype= int)

for n in range(r):

    H[n,:] *= 1.0 / (H[n,:].max())
    maxima, _ = scipy.signal.find_peaks(H[n,:], height=0.2, distance=18)
    localmax[n, 0:len(maxima)] = maxima
    

maximas = np.concatenate([localmax[0],localmax[1],localmax[2]])
maximas = maximas[maximas != 0]
maximas = np.sort(maximas)


for i in range(0,len(maximas)-1):
    if(maximas[i+1] <= maximas[i] + 10):
        idx1 = np.where(localmax == maximas[i])[0][0]
        idx2 = np.where(localmax == maximas[i+1])[0][0]

        if(H[idx1,maximas[i]] > H[idx2,maximas[i+1]]):
            localmax[idx2, np.where(localmax[idx2] == maximas[i+1])[0]] = 0
        else:
            localmax[idx1,np.where(localmax[idx1] == maximas[i])[0]] = 0



triggers = np.zeros((3,numframes),dtype=int)
temps = np.zeros((3,100), dtype= int)


for n in range(r):

    max = localmax[n][localmax[n] != 0]
    temps[n, 0:len(max)] = max
    H[n,:] *= 1.0 / (H[n,:].max())

    for i in range(numframes):
        if(i in max):
            triggers[n,i] = 1


snare, _ = librosa.load("Fakie Flip Snare.wav", sr=None) # Loading in audio file
kick, _ = librosa.load("Nollie Kick.wav", sr=None) # Loading in audio file
hihat, _ = librosa.load("Heel Flip Hat.wav", sr=None) # Loading in audio file



test = np.zeros((3,len(triggers[0]) + (len(triggers[0])-1)*(255)))
for n in range(3):
    test[n][::256] = triggers[n]


out = np.zeros(len(test[0])+93000)

for samp in range(len(test[0])):
    if(test[0,samp] == 1):
        out[samp:len(snare)+samp] = snare
    elif(test[1,samp] == 1):
        out[samp:len(kick)+samp] = kick
    elif(test[2,samp] == 1):
        out[samp:len(hihat)+samp] = hihat

if(x.size < out.size):
    out = out[0:len(x)]

print("Playing Audio")

while True:
    
    
    sd.play(out, 44100)
    time.sleep(len(out)/48000)
    sd.stop()

    check = input("Is this good? (Y/N)")
    if check == 'Y':
        input()
        break