# drumsep
A Convolutional Neural Network for drum signal separation from full mixes. 

For a quick demontration, do any of the following:
- watch this short video https://drive.google.com/file/d/1TXTkHynMJmXa_CawrM5GT_Lk6eah4BbE/view?usp=sharing
- open up the evaluate.ipynb notebook with an IPython kernel and play the audio in the cell outputs
- run the network on your own recording by: 1. cloning this repository on your machine 2. installing the required packages `pip install -r requirements.txt` 3. running the program `python3 evaluate.py <song_path> <audio_output_path>`

Method based on this this paper by K. W. E. Lin, B. Balamurali, E. Koh, S. Lui, and D. Herremans: https://arxiv.org/abs/1812.01278 

... and this tutorial by Ale Koretzky: https://towardsdatascience.com/audio-ai-isolating-vocals-from-stereo-music-using-convolutional-neural-networks-210532383785

Network trained on the MUSDB18-HQ dataset: https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems
