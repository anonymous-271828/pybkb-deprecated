# Learning the Finer Things: Bayesian Structure Learning at the Instantiation Level
This repository is the official implementation of [Learning the Finer Things](https://openreview.net/forum?id=tQQiKqGLK0g&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FConference%2FAuthors%23your-submissions)).

## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

If you wish to run BN or BKB learning from scratch then you will also need to install [pygobnilp](https://www.cs.york.ac.uk/aig/sw/gobnilp/) as well as obtain an academic liscence to [Gurobi](https://www.gurobi.com/) as this is the optimization suite that the GOBNILP backend uses. 

## Training
We have included pre-trained BKBs and BNs at the following anonymous [Google Drive link]() which can be downloaded directly our mounted in the cloud using Google Colab which will be discussed the Evaluation section.

If you wish to run training from scratch, you will need to ensure the above requirements are met, download the pre-processed KEEL datasets and/or TCGA breast cancer dataset from the following anonymous [Google Drive Link]() and run the following to scripts:
1. Navigate to nips_experiments
2. Run the following script to generate the scores for KEEL benchmarks:
```score
python3 run_keel_benchmark_scores.py
```
4. Upon completion of the KEEL score calculations, run the following script to learn the associated BKBs and BNs for KEEL:
```learn
python3 run_keel_benchmark_from_scores.py
```

To run the TCGA learning, first download the data as described above and run:
```learn_tcga
python3 run_tcga.py
```

## Evaluation
