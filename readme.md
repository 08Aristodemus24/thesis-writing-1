# This repository contains all generalized code snippets and templates relating to model experimentation, training, evaluation, testing, server-side loading, client-side requests, usage documentation, loaders, evaluators, visualizers, and preprocessor utilities, and the model architectures, figures, and final folder

# Requirements:
1. git
2. conda
3. python

# Source code usage
1. assuming git is installed clone repository by running `git clone https://github.com/08Aristodemus24/<repo name>`
2. assuming conda is also installed run `conda create -n <environment name e.g. some-environment-name> python=x.x.x`. Note python version should be `x.x.x` for the to be created conda environment to avoid dependency/package incompatibility.
3. run `conda activate <environment name used>` or `activate <environment name used>`.
4. run `conda list -e` to see list of installed packages. If pip is not yet installed run conda install pip, otherwise skip this step and move to step 5.
5. navigate to directory containing the `requirements.txt` file.
5. run `pip install -r requirements.txt` inside the directory containing the `requirements.txt` file
6. after installing packages/dependencies run `python index.py` while in this directory to run app locally

# App usage:
1. control panel of app will have ff. inputs: raw eda signals

# File structure:
```
|- client-side
    |- public
    |- src
        |- assets
            |- mediafiles
        |- boards
            |- *.png/jpg/jpeg/gig
        |- components
            |- *.svelte/jsx
        |- App.svelte/jsx
        |- index.css
        |- main.js
        |- vite-env.d.ts
    |- index.html
    |- package.json
    |- package-lock.json
    |- ...
|- server-side
    |- modelling
        |- data
        |- figures & images
            |- *.png/jpg/jpeg/gif
        |- final
            |- misc
            |- models
            |- weights
        |- metrics
            |- __init__.py
            |- custom.py
        |- models
            |- __init__.py
            |- arcs.py
        |- research papers & articles
            |- *.pdf
        |- saved
            |- misc
            |- models
            |- weights
        |- utilities
            |- __init__.py
            |- loaders.py
            |- preprocessors.py
            |- visualizers.py
        |- __init__.py
        |- experimentation.ipynb
        |- testing.ipynb
        |- training.ipynb
    |- static
        |- assets
            |- *.js
            |- *.css
        |- index.html
    |- index.py
    |- server.py
    |- requirements.txt
|- demo-video.mp5
|- .gitignore
|- readme.md
```

# Articles, Videos, Papers, Repositories:
1. multiple/ensemble model training: 
* https://www.geeksforgeeks.org/lazy-predict-library-in-python-for-machine-learning/
* https://medium.com/omics-diary/how-to-use-the-lazy-predict-library-to-select-the-best-machine-learning-model-65378bf4568e

2. evaluating ensemble models:

3. repositories:
* Hossains github profile: https://github.com/Md-Billal (confirmed thsi guys github profile and science direct profile both say university of connecticut and both have same names)
* Hossains UCOnn profile: https://biosignal.uconn.edu/person/md-hossain/
* Taylor et al. repo: https://github.com/MITMediaLabAffectiveComputing/eda-explorer
* Llanes-Jurado et al repo: https://github.com/ASAPLableni/EDABE_LSTM_1DCNN

# Insights:
1. WESAD dataset may contain raw eda signals that are labeled with stress or not stressed

# To do:

## artifact detection and correction:
* clone and review repo of Taylor et al.
* detect and correct the artifacts using LSTM-SVM and Llanes-Jurado et al. correction pipeline
* find dataset about stress detection with raw eda signals
* find out how autoregressive feature extraction from Hossain et al. works
* how to segment signals into 0.5s and 5s epochs/windows
* 

## SVM mechanism
* I need to learn how SVM can be implemented in tensorflow
* I need to learn also how SVM works explained by Andrew Ng

## stress detection:
* beacuse stress detection requires raw eda signals be corrected deconvolved into phasic components we will have to find a dataset that has raw eda signals labeled with stress or non stress and train a second model
* once the EDABE dataset has been denoised or the artifacts detected and subsequently corrected, we will have to use this second model trained on stress detection data to identify indeed what segments of our denoised EDABE dataset indicate a heightened level of stress
* annotation guidelines for stress detection dataset once EDABE dataset is cleaned
* 

## Task delegation
* work around the clock like madmen 30min on and then off then repeat
* Deseree and Johana on the paper
* me and Taline sa implementation

## Messaging Experts
* message llanes jurado et al taylor et al hossain et al.
* message armielyn
* message other GDSC leads
* message data science groups in facebook
* message alex gamboa

"Larry, Taline, Deseree, and Wana will graduaate in 2025"