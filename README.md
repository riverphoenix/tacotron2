# Tacotron 2

## Implementation of Tacotron 2

## Initial attempt

This is a tensorflow implementation of [NATURAL TTS SYNTHESIS BY CONDITIONING WAVENET ON MEL SPECTROGRAM PREDICTIONS](https://arxiv.org/pdf/1712.05884.pdf).

Initially I will use existing components from tacotron and other opensource implementations

## Usage

Main configuration is first to decide the 'run options' inside hyperparams.py
- test_graph = True # If you want to generate samples then set this to True
- include_dones = True # If you want to predict Done [end of generation] then set this to True
- train_form = 'Both' # 'Encoder', 'Converter', 'Both'  [Only train Encoder [main model], Converter [Griffin-Lim] or both]
- test_only = 0 # No training, just generate samples [integer on number of samples, must be used with pretrained [saved] model]
- print_shapes = True # Print shapes of tensors for debugging during generation of graph

First run 'python prepro.py' to generate the training data
- Requires all data in dataset folder under name provider by 'data' hyperparam
- All audio in wav folder
- metadata.csv file containg text in form "LJ001-0001|Printing, in the only sense |Printing, in the only sense"
- cmudict.dict.txt if you use hyperparam cmu=True

Then run 'python train.py' for the actual training/generation/loading of model/samples. Typical usages:
- ``` python train.py --log_dir=logs --log_name=test --data_paths=datasets/default --deltree=True ```, logs defines log output directory, log_name defines name of current run, data_paths the directory of the training data, deltree to delete folder
- ``` python train.py --log_dir=logs --log_name=test --data_paths=datasets/default --load_path=logs/test ```, load_path the folder to load previous trained model
- ``` python train.py --log_dir=logs --log_name=test --data_paths=datasets/default --load_path=logs/test --load_converter=logs/converter ```, load_converter the folder to load pretrained converter

Hyperparameters for training and testing:
- summary_interval = 1 # every X epochs generate summary
- test_interval = 3 # every X epochs generate audio sample
- checkpoint_interval = 1 # every X epochs save model (required for test_interval to be before every audio sample epooch)

For better training decouple the phase into 3 steps:
1. ``` python train.py --log_dir=logs --log_name=Encoder --data_paths=datasets/default --deltree=True ```, train the encoder alone. For this set [train_form='Encoder']
2. ``` python train.py --log_dir=logs --log_name=Converter --data_paths=datasets/default --deltree=True ```, train the converter alone. For this set [train_form='Converter']
3. ``` python train.py --log_dir=logs --log_name=Encoder --data_paths=datasets/default --load_path=logs/Encoder --load_converter=logs/Converter ```, Generate samples using trained econder and converter. For this set [train_form='Encoder' and 'test_only = 1']

### Results
To be posted in a few days (work in progress)

### ToDo
1. Replace the current Griffin-Lim converter with trained Wavenet vocoder (Ryuichi Yamamoto - https://github.com/r9y9/wavenet_vocoder)
2. Use better dataset (less noisy)

### Data

LJ Speech Dataset(https://keithito.com/LJ-Speech-Dataset)

### References

A lot of the base work has been taken from Kubyong Park's (kbpark.linguist@gmail.com) implementation of Deep Voice 3 (https://www.github.com/kyubyong/deepvoice3) and also Rayhane Mama's (https://www.linkedin.com/in/rayhane-mama/) implementation of Tacotron 2 (https://github.com/Rayhane-mamah/Tacotron-2) for the Attention Mechanism and wrapper

Feel free to reach out to me if you want to contribute (dimitris@rsquared.io)