import pyaudio
import librosa, librosa.display
import numpy as np
import scipy
import sounddevice as sd
from playsound import playsound
import time
from NMF_Function import myNMF

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# constants
CHUNK = 256                  # samples per frame
FORMAT = pyaudio.paFloat32     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
RECORD_SEC = 2

# pyaudio class instance
p = pyaudio.PyAudio()

# initialize stream object
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

## KICK
frames = [] # A python-list of chunks(numpy.ndarray)
while True:

    print('~~~~~~~~~~~~~~~~~~~~')
    print('Press Enter to begin recording Kick')
    input()

    
    time.sleep(.5)

    for _ in range(0, int(RATE / CHUNK * RECORD_SEC)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(np.fromstring(data, dtype=np.float32))

    
    playsound('record_end.wav')
    #Convert the list of numpy-arrays into a 1D array (column-wise)
    kick = np.hstack(frames)
    
    print('Playing back...')
    time.sleep(1)
    sd.play(kick, RATE)
    time.sleep(RECORD_SEC)
    sd.stop()

    
    check = input("Is this good? (Y/N)")
    if check == 'Y':
        print('\n')
        print('Kick recorded!')
        print('Press Enter to move onto Snare')
        print('~~~~~~~~~~~~~~~~~~~~')
        input()
        
        break
    else:
        frames = []
        print('\n\n')
        continue


## SNARE
frames = [] # A python-list of chunks(numpy.ndarray)
while True:

    print('~~~~~~~~~~~~~~~~~~~~')
    print('Press Enter to begin recording Snare')
    input()

    
    time.sleep(.5)

    for _ in range(0, int(RATE / CHUNK * RECORD_SEC)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(np.fromstring(data, dtype=np.float32))

    

    #Convert the list of numpy-arrays into a 1D array (column-wise)
    snare = np.hstack(frames)
    playsound('record_end.wav')
    print('Playing back...')
    time.sleep(1)
    sd.play(snare, RATE)
    time.sleep(RECORD_SEC)
    sd.stop()

    
    check = input("Is this good? (Y/N)")
    if check == 'Y':
        print('\n')
        print('Snare recorded!')
        print('Press Enter to move onto Hihat')
        print('~~~~~~~~~~~~~~~~~~~~')

        input()
        
        break
    else:
        frames = []
        print('\n\n')
        continue



## HIHAT
frames = [] # A python-list of chunks(numpy.ndarray)
while True:

    print('~~~~~~~~~~~~~~~~~~~~')
    print('Press Enter to begin recording Hihat')
    input()

    
    time.sleep(.5)

    for _ in range(0, int(RATE / CHUNK * RECORD_SEC)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(np.fromstring(data, dtype=np.float32))


    #Convert the list of numpy-arrays into a 1D array (column-wise)
    hihat = np.hstack(frames)
    playsound('record_end.wav')
    print('Playing back...')
    time.sleep(1)
    sd.play(hihat, RATE)
    time.sleep(RECORD_SEC)
    sd.stop()

    
    check = input("Is this good? (Y/N)")
    if check == 'Y':
        print('\n')
        print('Hihat recorded!')
        print('Press Enter to complete recording')
        print('~~~~~~~~~~~~~~~~~~~~')
        input()
        
        break
    else:
        frames = []
        print('\n\n')
        continue

print('Recording Samples')

frames = [] # A python-list of chunks(numpy.ndarray)
while True:

    print('~~~~~~~~~~~~~~~~~~~~')
    print('Press Enter to begin recording Beatboxing')
    input()

    
    time.sleep(.5)

    for _ in range(0, int(RATE / CHUNK * 10)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(np.fromstring(data, dtype=np.float32))

    
    playsound('record_end.wav')
    #Convert the list of numpy-arrays into a 1D array (column-wise)
    recording = np.hstack(frames)
    
    print('Playing back...')
    time.sleep(1)
    sd.play(recording, RATE)
    time.sleep(10)
    sd.stop()

    
    check = input("Is this good? (Y/N)")
    if check == 'Y':
        print('\n')
        print('Beatboxing recorded!')
        
        break
    else:
        frames = []
        print('\n\n')
        continue

# close stream
stream.stop_stream()
stream.close()

p.terminate()


# STFT Parameters
fftlen = 1024
hopsize = 256

# Training the dictionary components
print("Training Dictionary")

W = np.zeros((513,3)) # Dictionary Matrix

# Calculate the spectrogram
S = librosa.stft(kick,hop_length=hopsize,win_length=fftlen, window='hamming', n_fft=fftlen)
S_mag, S_phase = librosa.magphase(S)
S_db = librosa.amplitude_to_db(S_mag)

# NMF Parameters
r = 1
nIter = 75

[W_temp,H,KL] = myNMF(S_mag,r,nIter) # Process the Dictionary Matrix
W[:,0] = W_temp[:,0]

print(f'Component 0 Trained')


# Calculate the spectrogram
S = librosa.stft(snare,hop_length=hopsize,win_length=fftlen, window='hamming', n_fft=fftlen)
S_mag, S_phase = librosa.magphase(S)
S_db = librosa.amplitude_to_db(S_mag)

# NMF Parameters
r = 1
nIter = 75

[W_temp,H,KL] = myNMF(S_mag,r,nIter) # Process the Dictionary Matrix
W[:,1] = W_temp[:,0]

print(f'Component 1 Trained')


# Calculate the spectrogram
S = librosa.stft(hihat,hop_length=hopsize,win_length=fftlen, window='hamming', n_fft=fftlen)
S_mag, S_phase = librosa.magphase(S)
S_db = librosa.amplitude_to_db(S_mag)

# NMF Parameters
r = 1
nIter = 75

[W_temp,H,KL] = myNMF(S_mag,r,nIter) # Process the Dictionary Matrix
W[:,2] = W_temp[:,0]

print(f'Component 2 Trained')



# Loading in Test Audio File
x = recording

# Calculating the spectrogram of the audio file
S = librosa.stft(x,hop_length=hopsize,win_length=fftlen, window='hamming', n_fft=fftlen)
S_mag, S_phase = librosa.magphase(S)
S_db = librosa.amplitude_to_db(S_mag)

# NMF Parameters
r = 3
nIter = 75

[_,H,KL] = myNMF(S_mag,r,nIter,bUpdateW=0,initW=W) # Processing the audio file
print("NMF Completed")


numframes = np.floor(x.size/hopsize).astype(int) # calculating the number of total frames

localmax = np.zeros((3,100), dtype= int) # Variable for the local maxima of each component's activation matrix

for n in range(r):

    H[n,:] *= 1.0 / (H[n,:].max()) # Normalization

    maxima, _ = scipy.signal.find_peaks(H[n,:], height=0.2, distance=18) # Finding local maxima
    localmax[n, 0:len(maxima)] = maxima
    
# Organizing and sorting local maxima into one array
maximas = np.concatenate([localmax[0],localmax[1],localmax[2]])
maximas = maximas[maximas != 0]
maximas = np.sort(maximas)

# Detecting and processing duplicate or false maxima detection
for i in range(0,len(maximas)-1):

    # If any successive maxima are within a certain distance
    if(maximas[i+1] <= maximas[i] + 10): 
        try:
            idx1 = np.where(localmax == maximas[i])[0][0]
        except:

            idx1 = 0
        try:
            idx2 = np.where(localmax == maximas[i+1])[0][0]
        except:
            idx2 = 0
        

        # Check which maxima has a higher magnitude and delete the other false maxima
        if(H[idx1,maximas[i]] > H[idx2,maximas[i+1]]):
            localmax[idx2, np.where(localmax[idx2] == maximas[i+1])[0]] = 0
        else:
            localmax[idx1,np.where(localmax[idx1] == maximas[i])[0]] = 0



# Arrays for finding final triggers
triggers = np.zeros((3,numframes),dtype=int)
temps = np.zeros((3,100), dtype= int) 

for n in range(r):

    max = localmax[n][localmax[n] != 0] # Find all maxima not equal to 0
    temps[n, 0:len(max)] = max # Load into temp array

    for i in range(numframes): # Set impulse at maxima location
        if(i in max):
            triggers[n,i] = 1


snare, _ = librosa.load("Fakie Flip Snare.wav", sr=None) # Loading in audio file
kick, _ = librosa.load("Nollie Kick.wav", sr=None) # Loading in audio file
hihat, _ = librosa.load("Heel Flip Hat.wav", sr=None) # Loading in audio file



triggers_interp = np.zeros((3,len(triggers[0]) + (len(triggers[0])-1)*(255))) # Reconstructing the output through interpolation
for n in range(3):
    triggers_interp[n][::256] = triggers[n]

out = np.zeros(len(triggers_interp[0]) + RATE * RECORD_SEC) # Output with a buffer at the end for audio recording

# Check for trigger and load corresponding audio sample
for samp in range(len(triggers_interp[0])):
    if(triggers_interp[1,samp] == 1):
        out[samp:len(snare)+samp] = snare
    elif(triggers_interp[0,samp] == 1):
        out[samp:len(kick)+samp] = kick
    elif(triggers_interp[2,samp] == 1):
        out[samp:len(hihat)+samp] = hihat

if(x.size < out.size): # Match lengths of audio signals
    out = out[0:len(x)]

print("Playing Audio")

while True: # Play output 
    
    sd.play(out+x, RATE)
    time.sleep(len(out)/RATE)
    sd.stop()

    check = input("Is this good? (Y/N)")
    if check == 'Y':
        input()
        break