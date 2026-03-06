conda env create -f orca_train.yaml
conda activate develop

cd selene
python setup.py build_ext --inplace
python setup.py install