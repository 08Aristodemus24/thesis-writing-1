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
* Taylor et al. repo: https://github.com/MITMediaLabAffectiveComputing/eda-explorer
* Llanes-Jurado et al repo: https://github.com/ASAPLableni/EDABE_LSTM_1DCNN

4. annotation guidelines for peak detection indicating emotional arousal which may indicate stress
* https://edaguidelines.github.io/ref/
* https://imotions.com/blog/learning/research-fundamentals/eda-peak-detection/
* 

5. documentation on how to use ledalab which is a Matlab-based software for the analysis of skin conductance data (SC; i.e., EDA, GSR).
* can import various file formats (including BioPac, Biotrace, CassyLab, PortiLab, PsychLab, VarioPort, VisionAnalyzer, VitaPort) and provides many preprocessing functions.
* performs event-related analysis relative to events/marker and returns various parameters of phasic and tonic activity.
* can be used via an interactive GUI or in an efficient batch-mode via the Matlab command window.
* currently provides two EDA analysis methods:
(1) The Continuous Decomposition Analysis (CDA) performs a decomposition of SC data into continuous signals of phasic and tonic activity. This method takes advantage from retrieving the signal characteristics of the underlying sudomotor nerve activity (SNA). It is beneficial for all analyses aiming at unbiased scores of phasic and tonic activity.
(2) The Discrete Decomposition Analysis (DDA) perfroms a decomposition of SC data into distinct phasic components and a tonic component by means of Nonnegative Deconvolution. This method is especially advantageous for the study of the SCR shape.
* http://www.ledalab.de/

6. autoregressive modelling
* https://www.geeksforgeeks.org/autoregressive-ar-model-for-time-series-forecasting/

7. variable frequncy complex demodulation
* https://github.com/hasanmdabid/DA_Project/tree/c2e027108afba30a86768a9b3ffc7006393c09cd/BioVid

8. BioVid heat pain database from gouverneur et al. (2023) paper
https://www.nit.ovgu.de/nit/en/BioVid-p-1358.html

9. recursive feature elimination (RFE)
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
https://www.analyticsvidhya.com/blog/2021/04/backward-feature-elimination-and-its-implementation/

# Insights:
1. WESAD dataset may contain raw eda signals that are labeled with stress or not stressed
2. $AR(1)$ Model:
In the $AR(1)$ model, the current value depends only on the previous value.
It is expressed as: $Y_t = \beta + \theta_1Y_{t - 1} + \epsilon_t$
3. $AR(p)$ Model:
The general autoregressive model of order `p` includes `p` lagged values.
It is expressed as: $Y_t = \beta + \theta_1Y_{t - 1} + \theta_2Y_{t - 2} + \cdots + \theta_{p}Y_{t - p} + \epsilon_t$
4. what is rfe_loso mean in the context of artifact detection in eda signals or just signal processing with biosignal data from individual subjects in general?

- Recursive Feature Elimination (RFE) is a feature selection technique used to identify the most relevant features for a specific task (e.g., artifact detection). It works by iteratively removing the feature that contributes the least to the model's performance and retraining the model with the remaining features. This process continues until a desired number of features is reached or a stopping criterion is met.

- Leave-One-Subject-Out Cross-Validation (LOSO) is a cross-validation technique used to evaluate the generalizability of a model on unseen data. In LOSO, the model is trained on data from all subjects except one, and its performance is evaluated on the left-out subject's data. This process is repeated for all subjects, providing an estimate of how well the model generalizes to new individuals.

- Combining RFE and LOSO (rfe_loso)

- In the context of artifact detection, "rfe_loso" might indicate a two-step process:
Feature Selection with RFE: First, RFE is applied within LOSO cross-validation. For each fold (where one subject is left out), RFE selects the most relevant features for artifact detection using the training data from the remaining subjects. Artifact Detection with Selected Features: Finally, using the selected features from each LOSO fold, an artifact detection model is trained and applied to the left-out subject's data to identify potential artifacts.

- "rfe_loso" suggests a feature selection and model evaluation strategy that leverages both RFE and LOSO cross-validation for artifact detection or other biosignal processing tasks involving individual subjects. It emphasizes selecting features that are relevant across subjects while ensuring the model generalizes well to unseen data.

- so other than different hyperparams being tried k-times, we also try different features k-times

- class sklearn.feature_selection.RFE(estimator, *, n_features_to_select=None, step=1, verbose=0, importance_getter='auto') Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through any specific attribute or callable. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

however in our paper we will be using backward feature selection

# To do:

## artifact detection and correction:
* <s>clone and review repo of Taylor et al.</s>
* <s>how to segment signals into 0.5s and 5s epochs/windows</s>
* <s>decompose signal into phasic and tonic components</s>
* detect and correct the artifacts using LSTM-SVM and Llanes-Jurado et al. correction pipeline
* find dataset about stress detection with raw eda signals
* find out what BioVid and PainMoinit dataset from gouverneur et al. (2023) to trace the code they wrote
* review repo of gouverneur et al. (2023) since it details how autoregressive feature extraction, VFCDM from Hossain et al. (2022) works
* review repo of gouverneur et al. (2023) to learn how cvxEDA works
* <s>review repo of gouverneur et al. (2023) to learn how RFE-LOSO 5-fold cross validation works</s>




## SVM mechanism
* I need to learn how SVM can be implemented in tensorflow
* I need to learn also how SVM works explained by Andrew Ng

## stress detection:
* beacuse stress detection requires raw eda signals be corrected deconvolved into phasic components we will have to find a dataset that has raw eda signals labeled with stress or non stress and train a second model
* once the EDABE dataset has been denoised or the artifacts detected and subsequently corrected, we will have to use this second model trained on stress detection data to identify indeed what segments of our denoised EDABE dataset indicate a heightened level of stress
* annotation guidelines for stress detection dataset once EDABE dataset is cleaned
* review Smart Technologies for Long-Term Stress Monitoring at Work
* 

## Task delegation
* work around the clock like madmen 30min on and then off then repeat
* Deseree and Johana on the paper
* me and Taline sa implementation

## Messaging Experts
* <s>message llanes jurado et al</s>
- Jose Llanes-Jurado
email: jollaju101194@gmail.com, jllajur@upvnet.upv.es
- Lucía A. Carrasco-Ribelles
email: lcarrasco@idiapjgol.info
- Mariano Alcañiz
email: malcaniz@i3b.upv.es
- Emilio Soria-Olivas
email: emilio.soria@uv.es
- Javier Marín-Morales
email: jamarmo@i3b.upv.es

```
Student Inquiry on Wavelet Feature Extraction from EDA Signals in ""Automatic artifact recognition and correction for electrodermal activity based on LSTM-CNN models"" paper
```

```
Dear Dr. [name of researcher],

I hope this message finds you well. My name is Larry Miguel R. Cueva, and I am a student researcher at the Polytechnic University of the Philippines working on a project related to extracting wavelet features from EDA signals. I am writing to you today because I was very impressed by your work on this topic in the paper titled "Automatic artifact recognition and correction for electrodermal activity based on LSTM-CNN models".

I am particularly interested in the section where you describe your implementation of wavelet decomposition for feature extraction from EDA signals. I am currently trying to replicate this process in my own thesis using python and libraries like pywt but I face some challenges with regards to finding solutions to a particular problem in this task.

Specifically, I am facing difficulties with the Level 3 Haar wavelet decomposition using 128hz signals with uneven signal lengths after decomposition. I have been exploring potential solutions, but I would be incredibly grateful if you could offer some guidance on how you addressed these issues in your implementation.

I understand that you are as I'm sure very likely busy, as an already esteemed researcher such as yourself, but any insights you could share on your approach to wavelet feature extraction from EDA signals would be immensely helpful for our thesis given we are merely still but humble students learning what we can from this field. I am eager to learn and would be more than happy to schedule a brief call or exchange emails at your convenience to discuss this further.

Thank you for your time and consideration.

Sincerely,

Larry

Polytechnic University of the Philippines
```

* <s>message taylor et al.</s>
- Sara Taylor:
email: edaexplorer.reporting@gmail.com
- Natasha Jaques:
email: natashamjaques@gmail.com 
- Weixuan Chen
- Szymon Fedor
- Akane Sano
- Rosalind Picard

```
Student Inquiry on Wavelet Feature Extraction from EDA Signals in "Automatic identification of artifacts in electrodermal activity data" paper
```

```
Dear Dr. Taylor,

My name is Larry Miguel R. Cueva, and I am a student researcher at the Polytechnic University of the Philippines working on a project related to extracting wavelet features from EDA signals. I am writing to you today because I was very impressed by your work on this topic in the paper titled Automatic Identification of Artifacts in Electrodermal Activity Data.

I am particularly interested in the section where you describe your implementation of wavelet decomposition for feature extraction from EDA signals. I am currently trying to replicate this process in my own thesis using python and libraries like pywt but I face some challenges with regards to finding solutions to a particular problem in this task.

Specifically, I am facing difficulties with the Level 3 Haar wavelet decomposition using 128hz signals with uneven signal lengths after decomposition. I have been exploring potential solutions, but I would be incredibly grateful if you could offer some guidance on how you addressed these issues in your implementation.

I understand that you are as I'm sure very likely busy, as an already esteemed researcher such as yourself, but any insights you could share on your approach to wavelet feature extraction from EDA signals would be immensely helpful for our thesis given we are merely still but humble students learning what we can from this field. I am eager to learn and would be more than grateful to schedule a brief call or exchange emails at your utmost convenience of course to discuss this further.

Thank you for your time and consideration.

Sincerely,

Larry

Polytechnic University of the Philippines
```

```
Dear Dr. Jacques/Ms. Natasha,

My name is Larry Miguel R. Cueva, and I am a student researcher at the Polytechnic University of the Philippines working on a project related to extracting wavelet features from EDA signals. I am writing to you today because I was very impressed by your work on this topic in the paper titled Automatic Identification of Artifacts in Electrodermal Activity Data.

I am currently trying to replicate the methodology of your paper in my own thesis particularly with regards to the section where you describe your implementation of wavelet decomposition for feature extraction from EDA signals using python and libraries like pywt. Unfortunately, however I face some difficult challenges with regards to finding solutions to a particular problem in this task.

Specifically, I am facing difficulties with the Level 3 Haar wavelet decomposition using 128hz signals with uneven signal lengths after decomposition. I have been exploring potential solutions, but I would be incredibly grateful if you could offer some guidance on how you addressed these issues in your implementation.

I understand that you are as I'm sure very likely busy, as an already esteemed researcher such as yourself, but any insights you could share on your approach to wavelet feature extraction from EDA signals would be immensely helpful for our thesis given we are merely still but humble students learning what we can from this field. I am eager to learn and would be more than grateful to schedule a brief call or exchange emails at your utmost convenience of course to discuss this further.

Thank you for your time and consideration.

Sincerely,

Larry

Polytechnic University of the Philippines
```

* <s>message hossain et al. about whether we could have their code repository</s>
- Md-Billal Hossain:
github profile: https://github.com/Md-Billal (confirmed thsi guys github profile and science direct profile both say university of connecticut and both have same names)
UConn profile: https://biosignal.uconn.edu/person/md-hossain/
email: 	md.b.hossain@uconn.edu
- Hugo F. Posada-Quintero
email: hugo.posada-quintero@uconn.edu
- Youngsun Kong
email: youngsun.kong@uconn.edu
- Riley McNaboe
email: riley.mcnaboe@uconn.edu
- Ki H. Chon:
email: ki.chon@uconn.edu

```
Student Inquiry on Feature Extraction from EDA Signals in "Automatic Motion Artifact Detection in Electrodermal Activity Data Using Machine Learning" paper
```

```
Dear Dr. Hossain,

My name is Larry Miguel R. Cueva, and I am a student researcher at the Polytechnic University of the Philippines working on a project related to training a model for Artifact Detection from EDA signals. I am writing to you today because I was very impressed by your work on this topic in the paper titled Automatic Motion Artifact Detection in Electrodermal Activity Data Using Machine Learning.

I am currently trying to replicate the methodology of your paper in my own thesis particularly with regards to the section where you describe your implementation of variable frequency complex demodulation and autoregressive modelling for feature extraction from EDA signals using python. Unfortunately, however I face some difficult challenges with regards to finding solutions to a particular problem in this task.

Specifically, I am facing difficulties with extracting autoregressive features and also applying VFCDM to 128hz EDA signals. I have been exploring potential solutions, and looking up code how these are implemented but to no avail I am yet to find a resource that would give me the understanding to implement it in code. Given your I'm sure expertise in the topic and my interest in it I would be incredibly grateful if you could at your utmost convenience of course offer some guidance on how you addressed these issues in your implementation.

I understand that you are as I'm sure very likely busy, as an already esteemed researcher such as yourself, but any insights you could share on your approach to feature extraction from EDA signals would be immensely helpful for our thesis given we are merely still but humble students learning what we can from this field. I am eager to learn and would be more than grateful to schedule a brief call or exchange emails at again your utmost convenience to discuss this further.

Thank you for your time and consideration.

Sincerely,

Larry

Polytechnic University of the Philippines
```

* message armielyn
* message other GDSC leads
* message data science groups in facebook
* message alex gamboa

# To do now:
* <s>send emails to gc and ask them to send their respective emails</s>
* follow up the group to send their emails (pukpukin mo if you have to)
* <s>ask taline for paper review or if not I will alone</s>
* <s>send emails to every single one of those researchers</s>
* prepare for possible zoom/google meet with them by preparin jupyter notebook, asking about uneven length wavelet features, 
* <s>send email to hossain et al. </s>
* fill up and sign agreement form to access BioVid database data and send to sascha.gruss@uni-ulm.de

"Larry, Taline, Deseree, and Wana will graduaate in 2025"