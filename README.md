# Overview

`mess-rnn` is a set of tools used for training a recurrent neural network based
on behavioral analysis of malware.

The tools interact with MESS system, which is used for running malicious
software in a sandboxed worker VM, while collecting data about behaviour of all
running processes using Process Monitor. The data is then preprocessed and used
for training a neural network, with thanks to Tensorflow and TFlearn frameworks. 

# Installation

1. Setup a Python virtualenv
```
virtualenv mess-rnn-venv -p python3
```

2. Enter the virtualenv
```
source mess-rnn-venv/bin/activate
```

3. Install requirements
```
cd mess-rnn
pip3 install -r requirements.txt
```

# Workspace

Project assumes specific workspace layout. It can be customized in
`mess-rnn.cfg`. If it is not stated otherwise, all input paths passed to tools in this repo are
relative to corresponding directories specified in the config. This makes it so that operating on trees of directories containig different
samples is easy.

# Usage

The project contains a few Python3 scripts. Together, they form a pipeline:
from data collection to testing the model. 


## analyze.py

Schedules analysis of executable sample files in MESS sandbox and
collects results. Multiple workers can be running at the same time.

### Toolkit

Toolkit is a `.zip` file sent to MESS worker for each analysis. It contains
PowerShell scripts, which, based on their filename, will be called before, 
after, or upon restart of the worker VM.

A special `after.ps1` script is responsible for converting `.pml` files (raw
output of Process Monitor) to `.csv`.

### Input directory structure

As was mentioned, a specific structure is assumed, where each sub directory
should contain the malicious samples as well as a `.txt` file describing them.
For example:
```
/data/Malware/Locky/
|-- 2016.10
|   |-- 65rfgb.exe
|   |-- 98h86f.exe
|   |-- Samples_description.txt
|   |-- erg7cbr.exe
|   |-- jhg45s-2.exe
|   `-- jhg45s.exe
|-- 2016.11
|   |-- 43ftybb8.exe
|   |-- 76vvyt.exe
|   |-- 87yfhc-2.exe
|   |-- 87yfhc.exe
|   |-- Samples_description.txt
|   `-- kjg56f7.exe
```

The `.txt` file structure is also specific. Samples should be grouped using the 
`EntryFunction:` keyword. For example:
```
EntryFunction: may
878hf33f34f.exe
878hf33f34f-2.exe

EntryFunction: total
6v5r7thh.exe
```

If a sample is just a simple `.exe` without an entry function (it runs 
automatically), then it doesn't need to be listed in such a way.

### Usage

If everything is configured correctly, using the script is as simple as calling
```
python3 analyze.py --dir Locky/2016.12
```

Whole subdirectory can be specified. This directory must be relative to the
input directory specified in the configuration file.

## preprocess.py

Unpacks zipped results from MESS analysis. Doing so, updates the dictionary
of encountered function calls and saves their encoded sequence for each
PID, dividing them into malevolent and benevolent.

This tool assumes, that `.zip` files are present from the `analyze.py`.

### Usage
```
python3 preprocess.py --dir Locky/2016.12
```

## bundle.py

Takes encoded history of each process in specified subdirectory and outputs
numpy datasets easily digestible in Tensorflow. Also, splits those datasets
into train and tests subsets.

### Usage
```
python3 bundle.py --dir Locky/2016.12 --name bundle_name
```

One can create any dataset they want by moving `.txt` files from `preprocess.py`
to a subdirectory, and then calling this script with that subdir as argument.
Bundle will be created recursively of all encodings under that subdir.

## rnn.py

Invokes tensorflow neural network. This script can be used to train a network,
load weights from previously trained network or just test it on some data.

### Training
```
python3 rnn.py --bundle my_network \
               --train
```

### Training on a previously trained model
```
python3 rnn.py --bundle my_network \
               --train \
               --pretrained 2017-11-17_19-37-05_my_network9971
```

### Testing on a previously trained model
```
python3 rnn.py --bundle my_network \
               --pretrained 2017-11-17_19-37-05_my_network9971
```

### Hyperparameters

Network and learing hyperparameters can be also set from command line.
For more information, see 
```
python3 rnn.py --help
```

# Other info

Tensorboard can be used to analyze the learning process:
```
tensorboard --logdir $tensorboard_dir_from_cfg
```
