import sys
import librosa
import librosa.output
import tensorflow
import numpy as np
from tensorflow.keras.models import model_from_json

def process_song(song, hop_length=512, n_fft=1024, context_size=25):
	"""
	Preprocesses one song and creates x-frames with associated y-labels in the target directory
	
	parameters:
		song: (ndarray) audio to be processed
		hop_length, n_fft, context_size: preprocessing parameters pertaining to the STFT spectrograms; make sure 
		they are the same as the ones used in training
	"""
	
	mix_spec = np.abs(librosa.stft(song, hop_length=hop_length, n_fft=n_fft))
	
	n_bins, n_frames = mix_spec.shape
	
	frames = []
	
	for i in range(n_frames):
		# container for one image of size n_bins, context_size
		x = np.zeros(shape=(n_bins, context_size))
		
		for j in range(context_size):
			curr_idx = i - context_size//2 + j
			
			# if current index out of range, leave 0s as padding
			if curr_idx < 0:
				continue
			elif curr_idx >= n_frames:
				break
				
			else:
				x[:, j] = mix_spec[:, curr_idx]
		
		frames.append(x)
			
	return np.expand_dims(np.asarray(frames), axis=-1)

def evaluate_song(song, model):
	"""
	Convenience method for quick application of the drum separation network.
	
	parameters:
		song: (ndarray) audio to be processed
		model: trained Keras model
	"""
	
	song_data = process_song(song)
	ibm = model.predict(song_data)
	ibm = ibm.T
	
	mixspec = librosa.stft(song, hop_length=512, n_fft=1024)
	reconst = librosa.istft(mixspec * ibm, hop_length=512)
	
	return reconst

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: %s <input filename> <output filename>" % (sys.argv[0]))
		exit()

	infile = sys.argv[1]
	outfile = sys.argv[2]

	# load user specified song
	song, sr = librosa.load(infile, sr=22050)

	# load and compile model
	model_path = "trained_model/"
	model_name = "drumsep_full"

	json_file = open(model_path + model_name + ".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights(model_path + model_name + ".h5")
	print("Loaded model %s from disk" % model_name)

	precision = tensorflow.keras.metrics.Precision()
	recall = tensorflow.keras.metrics.Recall()
	model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=[precision, recall])

	# apply trained network to song
	result = evaluate_song(song, model)

	# write file to user specified path
	librosa.output.write_wav(outfile, result, sr=22050, norm=False)
	print("\nProcessed %s successfully. Written result to %s.\n" % (infile, outfile))
