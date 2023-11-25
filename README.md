# drumsep
A Convolutional Neural Network for drum signal separation from full mixes. 

For a quick demontration, do any of the following:
- run the network on your own recording by:<br> 1. cloning this repository on your machine<br> 2. in a python3.7 environment, installing the required packages `pip install -r requirements.txt`<br> 3. running the program on your own WAVE or mp3 file `python3 evaluate.py <input_audio_path> <output_audio_path>`; or using the provided file snippet.wav (which was not used during training)
- watch this short video https://drive.google.com/file/d/1XmyjkJd7u3MMCC3NKeiOHhJ2nZe7ohR_/view?usp=drive_link
- open up the evaluate.ipynb notebook with an IPython kernel and play the audio in the cell outputs

Method based on this this paper by K. W. E. Lin, B. Balamurali, E. Koh, S. Lui, and D. Herremans: https://arxiv.org/abs/1812.01278 

... and this tutorial by Ale Koretzky: https://towardsdatascience.com/audio-ai-isolating-vocals-from-stereo-music-using-convolutional-neural-networks-210532383785

Network trained on the MUSDB18-HQ dataset: https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav
