"""
Marius Orehovschi
Jan 2021
Convolutional neural network for drum signal separation.

This file contains the data preprocessing functions (using the MUSDB18HQ dataset). Takes Short-Time Fourier 
Transform spectrograms of full mixes and frames them with context_size=25 frames. Then takes the STFT of drum
and non-drum audio separately and uses them to calculate the Ideal Binary Mask (IBM), which indicates 'drum' ('1')
frequency bins and 'non-drum' ('0') frequency bins
"""

import numpy as np
import librosa
import librosa.display
import os
from joblib import Parallel, delayed
import time

# where audio recordings are stored
source_dir = "/storage/moreho21/musdb18hq/raw_train/"

# remove useless macOS file
songs = os.listdir(source_dir)
if ".DS_Store" in songs:
    songs.remove(".DS_Store")

# where corresponding STFT spectrograms will be stored   
target_dir = "/storage/moreho21/musdb18hq/preprocessed_data/"

def process_song(source_dir, songname, target_dir, hop_length=512, n_fft=1024, context_size=25):
    """
    Preprocesses one song and creates x-frames with associated y-labels in the target directory.

    parameters:
        source_dir: (str) path to the source directory (MUSDB18HQ or a dataset with similar structure)
        songname: (str) name of audio recording to process
        target_dir: (str) path to where the data is to be stored
        hop_length: (int) length (in audio samples) of hop between STFT frames 
        n_fft: (int) length of the STFT context window
        context_size: (int) size of the window fed into CNN (corresponding to one timestep in the frequency domain)
    """
    
    # combine all the drumless tracks into one
    melo, sr = librosa.load(source_dir + songname + "/vocals.wav")
    melo += librosa.load(source_dir + songname + "/other.wav")[0]
    melo += librosa.load(source_dir + songname + "/bass.wav")[0]
    
    # drum track
    drum, sr = librosa.load(source_dir + songname + "/drums.wav")
    
    # mixture track
    mix, sr = librosa.load(source_dir + songname + "/mixture.wav")
    
    # take spectrograms of the 3 tracks
    melo_spec = np.abs(librosa.stft(melo, hop_length=hop_length, n_fft=n_fft))
    drum_spec = np.abs(librosa.stft(drum, hop_length=hop_length, n_fft=n_fft))
    mix_spec = np.abs(librosa.stft(mix, hop_length=hop_length, n_fft=n_fft))
    
    n_bins, n_frames = melo_spec.shape
    
    # container for frame names and associated labels
    fnames = []
    
    # 
    for i in range(n_frames):
        # container for one image of size n_bins, context_size
        x = np.zeros(shape=(n_bins, context_size))
        
        # frame each STFT time step with context_size//2 before and after (pad with 0s at the edges)
        for j in range(context_size):
            curr_idx = i - context_size//2 + j
            
            # if current index out of range, leave 0s as padding
            if curr_idx < 0:
                continue
            elif curr_idx >= n_frames:
                break
                
            else:
                x[:, j] = mix_spec[:, curr_idx]
        
        # save the current x frame
        xfname = target_dir + "x/%s_%d.npy" % (songname, i)
        np.save(xfname, x)
        
        # calculate the IBM for the current x frame
        y = drum_spec[:, i] - melo_spec[:, i]
        y = np.where(y > 0, 1, 0)
        
        # save the IBM
        yfname = target_dir + "y/%s_%d.npy" % (songname, i)
        np.save(yfname, y)
        
        fnames.append((xfname, yfname))
    
    # save the array of x-y filename associations as a ndarray        
    fnames = np.asarray(fnames)
    np.save(target_dir + "%s_fnames" % songname, fnames)

# process all directories separately in parallel
init_time = time.time()
Parallel(n_jobs=-1, verbose=1)(delayed(process_song)(source_dir, name, target_dir) for name in songs)
processing_time = time.time()-init_time
print("Time taken: %.2f seconds." % processing_time)

# take each individual song's filenames and combine them into one big array
targetdir_contents = os.listdir(target_dir)
all_fnames = []

for item in targetdir_contents:
    if item.endswith(".npy") and (item != "fnames.npy"):
        all_fnames.append(np.load(target_dir + item))
        
all_fnames = np.vstack(all_fnames)
np.save(target_dir + "fnames", all_fnames)

# remove the individual song filename dataframes
for item in targetdir_contents:
    if item.endswith(".npy") and (item != "fnames.npy"):
        os.unlink(target_dir + item)
