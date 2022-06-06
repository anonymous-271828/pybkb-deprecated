# Learning the Finer Things: Bayesian Structure Learning at the Instantiation Level
This repository is the official implementation of [Learning the Finer Things](https://openreview.net/forum?id=tQQiKqGLK0g&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FConference%2FAuthors%23your-submissions)).

## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

If you wish to run BN or BKB learning from scratch then you will also need to install [pygobnilp](https://www.cs.york.ac.uk/aig/sw/gobnilp/) as well as obtain an academic liscence to [Gurobi](https://www.gurobi.com/) as this is the optimization suite that the GOBNILP backend uses. 

## Training
We have included pre-trained BKBs and BNs at the following anonymous Zenodo repository which can be downloaded directly our mounted in the cloud using Google Colab which will be discussed the Evaluation section.

If you wish to run training from scratch, you will need to ensure the above requirements are met, download the pre-processed KEEL datasets and/or TCGA breast cancer dataset from the following anonymous links: [Preprocessed KEEL dataset](https://zenodo.org/record/6580480#.Yo-2IlRBxPY) and [Preprocessed TCGA breast cancer dataset](https://zenodo.org/record/6584753#.Yo-1g1RBxPY). Then run the following to scripts:
1. Navigate to nips_experiments
2. Run the following script to generate the scores for KEEL benchmarks (Optionally, pass a dataset name with --dataset_name=dataset otherwises the script will run through all the datasets):
```score
python3 run_keel_benchmark_scores.py /path/to/datasets /path/to/save/scores
```
4. Upon completion of the KEEL score calculations, run the following script to learn the associated BKBs and BNs for KEEL:
```learn
python3 run_keel_benchmark_from_scores.py /path/to/datasets /path/to/save/results /path/to/scores --palim=palim --dataset_name=dataset_name
```

To run the TCGA learning, first download the data as described above and run:
```learn_tcga
python3 run_tcga.py /path/to/dataset /path/to/save/results
```

## Evaluation
Evaluation of BKBs and BNs are best done in the supplied Jupyter Notebooks. First, you should download the pre-learned BKBs and BNs hosted on [Zenodo](https://zenodo.org/record/6577348#.Yp3kr-xBxPY). You will then need to unzip these files and follow the instructions in the corresponding notebooks:
- [KEEL Benchmark Analysis](https://github.com/anonymous-271828/pybkb/blob/master/notebooks/Keel%20Benchmark%20Analysis.ipynb)
- [TCGA Analysis](https://github.com/anonymous-271828/pybkb/blob/master/notebooks/TCGA%20Analysis.ipynb)

### Running the Notebooks
We recommend setting up a python virtual environment to run all jupyter notebooks locally.

1) Clone this repo and create a virtual environment with python 3.8 or above.
``` bash
python3.8 -m venv pybkb_venv
. pybkb_venv/bin/activate
```

2) Run the included setup.py with
``` bash
python3 setup.py install
```

3) Make sure to install pygobnilp with pip or clone the pygobnilp repository.
4) Install your virtual enivronment into the jupyter notebook kernels
``` bash
python3 -m ipykernel install --user --name=pybkb_venv
``` 
5) Install notebook extensions for ipywidgets with
``` bash
jupyter nbextension enable --py widgetsnbextension
```
6) Make sure you have downloaded the pre-learned BKBs from Zenodo [here](https://zenodo.org/record/6577348#.Yp3kr-xBxPY) and have unzipped these files locally.
7) Launch the jupyter with
``` bash
jupyter notebook
```
8) Open the respective notebook, i.e. KEEL Benchmark Analysis or TCGA Analysis, and follow the notebook's instructions to evaluate BNs and BKBs to your liking. 
