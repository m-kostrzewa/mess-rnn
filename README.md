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
`mess-rnn.cfg`.

If it is not stated otherwise, all input paths passed to tools in this repo are
relative to `input_base_dir`, and output files are relative to
`output_base_dir`.

This makes it so that operating on trees of directories containig different
samples is easy.

# Usage

1. Analyze samples in some subdir
```
python3 analyze.py --dir Locky/2016.12
```

2. Preprocess raw .zip files to integer encodings of operations
```
python3 preprocess.py --dir Locky/2016.12
```

3. Combine encodings into train and test sets (bundles)
```
python3 bundle.py --dir Locky/2016.12 --name bundle_name
```

4. Train the model
```
python3 rnn.py --name bundle_name --train
```

5. Run tensorboard to analyze learning process
```
tensorboard --logdir $tensorboard_dir_from_cfg
```

6. You may train a previously trained model further
```
python3 rnn.py --bundle abcd --train --pretrained 2017-11-17_19-37-05_abcd9971
```

7. Or you may just test on the input bundle, using a pretrained model
```
python3 rnn.py --bundle abcd --train --pretrained 2017-11-17_19-37-05_abcd9971
```