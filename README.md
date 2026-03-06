Instructions to start training Orca:
* Clone this repo locally using `git clone https://github.com/neevshaw/Orca-Train.git`
* Go to the [4DN H1-hESC file](https://data.4dnucleome.org/files-processed/4DNFI2TK7L2F/) and download it. Move the file into `orca/resources` folder.
* Install Miniconda with `install_conda.sh`
* Run `setup_env.sh` to install necessary packages
* Run `setup_data.sh` to unzip data and create genome Memory Map
* Run `train.sh` to start the training script.