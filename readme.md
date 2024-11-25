# test commands:
* `python tuning_dl.py -m lstm-cnn -pl jurado -lr 5e-5 --mode tuning`
* `python tuning_dl.py -m lstm-svm -pl cueva -lr 1e-3 --batch_size 1024 --mode tuning`
* `python tuning_ml.py -m lr -pl hossain --n_rows_to_sample 5000 --mode tuning`

* `python tuning_dl.py -m lstm-svm -pl cueva -lr 1e-3 --batch_size 1024 --mode training --hyper_param_list window_size_640 n_a_16 drop_prob_0.05 C_10 gamma_0.5 units_10`: command for preliminary modelling of proposed model
* `python tuning_dl.py -m lstm-svm -pl cueva -lr 75e-4 --batch_size 1024 --mode training --hyper_param_list window_size_640 n_a_16 drop_prob_0.05 C_1 gamma_0.1 units_10`
* `python tuning_dl.py -m lstm-svm -pl cueva -lr 75e-4 --batch_size 1024 --mode training --hyper_param_list window_size_640 n_a_16 drop_prob_0.05 C_0.7 gamma_0.5 units_10`

* `python tuning_dl.py -m lstm-fe -pl cueva -lr 1e-3 --batch_size 1024 --mode training --hyper_param_list window_size_640 n_a_32 drop_prob_0.3 lamb_da_0.1`
* `python tuning_dl.py -m lstm-cnn -pl jurado -lr 5e-5 --mode training --hyper_param_list window_size_640 n_a_16 drop_prob_0.05`

* `python tuning_ml.py -m svm -pl cueva_second_phase --n_features_to_select 70 --n_rows_to_sample 5000 --mode tuning`
* `python tuning_ml.py -m svm -pl taylor --n_rows_to_sample 5000 --mode tuning`

* `python tuning_ml.py -m svm -pl cueva_second_phase --mode training --hyper_param_list C_10 gamma_0.1 probability_True kernel_rbf`
* `python tuning_ml.py -m svm -pl cueva_second_phase --mode training --hyper_param_list C_10 gamma_0.1 probability_True kernel_poly degree_3`
* `python tuning_ml.py -m svm -pl cueva_second_phase --mode training --hyper_param_list C_10 gamma_0.1 probability_True kernel_rbf --inc_class_weight`
* `python tuning_ml.py -m svm -pl cueva_second_phase --mode training --hyper_param_list C_10 gamma_0.1 probability_True --inc_class_weight --comp_type 2.6`
* `python tuning_ml.py -m svm -pl hossain --mode training --hyper_param_list C_10 gamma_0.1 probability_True`
* `python tuning_ml.py -m svm -pl taylor --mode training --hyper_param_list C_10 gamma_0.1 probability_True`

* `scp -r -i C:/Users/LARRY/.ssh/id_rsa "C:/Users/LARRY/Documents/Scripts/thesis-writing-1/server-side/modelling/data/Artifact Detection Data/" michael.cueva@202.90.149.55:/home/michael.cueva/scratch1/thesis-writing-1/server-side/modelling/data/`
* `scp -r -i C:/Users/LARRY/.ssh/id_rsa michael.cueva@202.90.149.55:/home/michael.cueva/scratch1/thesis-writing-1/server-side/modelling/saved/models/taylor_svm_clf.pkl C:/Users/LARRY/documents/scripts/thesis-writing-1/server-side/modelling/saved/models/`

# This repository contains all generalized code snippets and templates relating to model experimentation, training, evaluation, testing, server-side loading, client-side requests, usage documentation, loaders, evaluators, visualizers, and preprocessor utilities, and the model architectures, figures, and final folder

# Requirements:
1. git
2. conda
3. python

# Source code usage
1. assuming git is installed clone repository by running `git clone https://github.com/08Aristodemus24/thesis-writing-1`
2. assuming conda is also installed run `conda create -n <environment name e.g. thesis-writing-1> python=3.12.3`. Note python version should be `3.12.3` for the to be created conda environment to avoid dependency/package incompatibility.
3. run `conda activate thesis-writing-1` or `activate thesis-writing-1`.
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
            |- Artifact Detection Data
                |- test
                    |- *_features.csv
                    |- *_labels.csv
                |- train
                    |- *_features.csv
                    |- *_labels.csv
                |- reduced_cueva_second_phase_svm_feature_set1.txt
                |- reduced_cueva_second_phase_svm_feature_set.txt
                |- hossain_feature_set.txt
                |- reduced_hossain_lr_feature_set.txt
                |- reduced_hossain_gbt_feature_set.txt
                |- reduced_hossain_svm_feature_set.txt
                |- taylor_feature_set.txt
                |- reduced_taylor_lr_feature_set.txt
                |- reduced_taylor_rf_feature_set.txt
                |- reduced_taylor_svm_feature_set.txt
            |- Electrodermal Activity artifact correction BEnchmark (EDABE)
                |- Train
                    |- *.csv
                |- Test
                    |- *.csv
            |- Hybrid Artifact Detection Data
                |- train
                    |- *_hof.csv
                    |- *_lof.csv
                    |- *_labels.csv
                |- test
                    |- *_hof.csv
                    |- *_lof.csv
                    |- *_labels.csv
                |- dummy.txt
            |- Hosseini_Stress_Dataset
            |- Stress Detection Features
            |- dummy.txt
            |- EDABE dataset.zip
            |- Stress_dataset.zip
        |- figures & images
            |- *.png/jpg/jpeg/gif
        |- models
            |- __init__.py
            |- cueva.py
            |- llanes_jurado.py
        |- results
            |- all_models_results.json
            |- hossain_gbt_results.json
            |- hossain_lr_results.json
            |- hossain_svm_results.json
            |- taylor_rf_results.json
            |- taylor_lr_results.json
            |- taylor_svm_results.json
            |- pqbqpr_expert2_corrected.csv
        |- saved
            |- misc
                |- cueva_lstm-fe_meta_data.json
                |- jurado_lstm-cnn_meta_data.json
                |- hossain_lr_scaler.pkl
                |- hossain_svm_scaler.pkl
                |- hossain_gbt_scaler.pkl
                |- xgb_scaler.pkl
                |- dummy.pkl
            |- models
                |- cueva_second_phase_svm_clf1.pkl
                |- cueva_second_phase_svm_clf.pkl
                |- hossain_lr_clf.pkl
                |- hossain_svm_clf.pkl
                |- hossain_gbt_clf.pkl
                |- taylor_lr_clf.pkl
                |- taylor_svm_clf.pkl
                |- taylor_rf_clf.pkl
                |- stress_detector.pkl
                |- dummy.pkl
            |- weights
                |- *.weights.h5
        |- utilities
            |- __init__.py
            |- loaders.py
            |- preprocessors.py
            |- visualizers.py
            |- evaluators.py
            |- feature_extractors.py
            |- stress_feature_extractors.py
        |- __init__.py
        |- experimentation.ipynb
        |- feature_engineering.ipynb
        |- data_analysis.ipynb
        |- summarization.ipynb
        |- evaluation.ipynb
        |- visualization.ipynb
        |- stress_detection.py
        |- tuning_ml.py
        |- tuning_dl.py
        |- *.sbatch
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

10. implementing radial basis function in tensorflow
https://www.kaggle.com/code/residentmario/radial-basis-networks-and-custom-keras-layers

11. using d3.js with react for visualizing line charts pie charts csv data etc.
https://d3-graph-gallery.com/graph/line_color_gradient_svg.html
https://www.digitalocean.com/community/tutorials/svg-svg-viewbox

12. https://www.neurology.org/doi/pdfdirect/10.1212/WNL.39.6.868
hybrid, long, short, term, memory, support, vector, machine, for, detecting recognizing, artifacts, in electrodermal activity data or galvanic skin response data signals, for stress detection recognition

13. why val loss of 0 may be appearing in custom tensorflow model in the test_step() method





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

5. The Dice Coefficient (DSC) is a similarity metric that ranges from 0 to 1.   

A DSC of 1 indicates perfect overlap between the predicted and true labels.   
A DSC of 0 indicates no overlap at all.
Unlike accuracy, which can be misleading in imbalanced datasets, DSC focuses specifically on the overlap between the positive classes. This makes it a more robust metric for evaluating binary classification models, especially in medical image segmentation and other applications where the positive class is relatively rare.

Key Points about DSC
Higher DSC values are better. The closer the value is to 1, the better the model's performance.
DSC is more sensitive to false negatives and false positives than accuracy, making it a good choice for tasks where minimizing these errors is crucial.
DSC can be used in conjunction with other metrics like accuracy, precision, recall, and F1-score to get a more comprehensive evaluation of a model's performance.
Example Interpretation
DSC = 0.95: This indicates a very high degree of overlap between the predicted and true positive labels, suggesting excellent model performance.
DSC = 0.50: This suggests moderate overlap, indicating room for improvement in the model's ability to correctly identify positive cases.
DSC = 0.10: This indicates very low overlap, suggesting poor model performance in identifying positive cases.
In summary, DSC is a valuable metric for assessing the performance of binary classification models, especially when dealing with imbalanced datasets or tasks where correctly identifying positive cases is critical.

The minimum value that the dice can take is 0 , which is when there is no intersection between the predicted mask and the ground truth.
The maximum value that the dice can take is 1 , which means the prediction is 99% correct. For that we will have the intersection equal to A or B (the prediction mask or the ground truth) because they are the same
You remember that we said that the best values of the dice are the values that are near to 1
So we can do a small equation using the Dice coefficient to get small values instead of values near to 1
we can conclude that when the dice value will go up then the dice loss will go down, and when we get the max value of the dice then we will get 0 in the loss which means that the model is perfect (probably)

6. testing on 500 rows of all subjects features and labels will likely produce imbalance of classes for each subject so it is imperative that I need to train on all subjects
https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
As mentioned in the comments, some labels in y_test don't appear in y_pred. Specifically in this case, label '2' is never predicted:
y_test = [0, 0, 0, 0, 0] y_pred = [1, 0, 0, 0, 0]

7. below is how Agarap (2017) utilized the output of the lstm cell and extract the last hidden state by using `tf.gather()` and then using this as the input feature for the SVM model
```
>>> import tensorflow as tf
>>> import numpy as np
>>>
>>> # 5000 is m number of training examples, 21 is T_x number of timesteps, and 256 is n_a number of hidden state neurons 
>>> lstm_out = tf.random.uniform(shape=(5000, 21, 256))
>>> lstm_out
<tf.Tensor: shape=(5000, 21, 256), dtype=float32, numpy=
array([[[0.28524446, 0.6920228 , 0.5331818 , ..., 0.33454227,
         0.9956434 , 0.46298552],
        [0.26323366, 0.39587176, 0.8516718 , ..., 0.42707026,
         0.02075613, 0.8760253 ],
        [0.98651373, 0.4172057 , 0.14282095, ..., 0.3269347 ,
         0.4674852 , 0.7167444 ],
        ...,
        [0.3596145 , 0.59132063, 0.6778234 , ..., 0.08824635,
         0.7366158 , 0.44244516],
        [0.5485084 , 0.67758715, 0.59662616, ..., 0.36997724,
         0.45383978, 0.5849253 ],
        [0.5911144 , 0.6347213 , 0.14406228, ..., 0.35377884,
         0.20562398, 0.54268634]],

       ...,

       [[0.5911144 , 0.6347213 , 0.14406228, ..., 0.35377884,
         0.20562398, 0.54268634],
        [0.81471217, 0.77297723, 0.21489036, ..., 0.75407195,
         0.5278652 , 0.74971557],
        [0.02764094, 0.6419848 , 0.8546988 , ..., 0.47864842,
         0.9153811 , 0.3288567 ],
        ...,
        [0.8812096 , 0.74812543, 0.90431917, ..., 0.4562235 ,
         0.9937532 , 0.6019323 ],
        [0.3773731 , 0.32255256, 0.45916831, ..., 0.0901674 ,
         0.4323442 , 0.23069155],
        [0.94965756, 0.8568523 , 0.06734467, ..., 0.606243  ,
         0.04353523, 0.79850745]]], dtype=float32)>
>>>
>>> lstm_out_t[20]
<tf.Tensor: shape=(5000, 256), dtype=float32, numpy=
array([[0.5911144 , 0.6347213 , 0.14406228, ..., 0.35377884, 0.20562398,
        0.54268634],
       [0.81471217, 0.77297723, 0.21489036, ..., 0.75407195, 0.5278652 ,
        0.74971557],
       [0.02764094, 0.6419848 , 0.8546988 , ..., 0.47864842, 0.9153811 ,
        0.3288567 ],
       ...,
       [0.8812096 , 0.74812543, 0.90431917, ..., 0.4562235 , 0.9937532 ,
        0.6019323 ],
       [0.3773731 , 0.32255256, 0.45916831, ..., 0.0901674 , 0.4323442 ,
        0.23069155],
       [0.94965756, 0.8568523 , 0.06734467, ..., 0.606243  , 0.04353523,
        0.79850745]], dtype=float32)>
>>>
>>> last_hidden_state = tf.gather(lstm_out_t, indices=lstm_out_t.shape[0] - 1)
>>> last_hidden_state
<tf.Tensor: shape=(5000, 256), dtype=float32, numpy=
array([[0.5911144 , 0.6347213 , 0.14406228, ..., 0.35377884, 0.20562398,
        0.54268634],
       [0.81471217, 0.77297723, 0.21489036, ..., 0.75407195, 0.5278652 ,
        0.74971557],
       [0.02764094, 0.6419848 , 0.8546988 , ..., 0.47864842, 0.9153811 ,
        0.3288567 ],
       ...,
       [0.8812096 , 0.74812543, 0.90431917, ..., 0.4562235 , 0.9937532 ,
        0.6019323 ],
       [0.3773731 , 0.32255256, 0.45916831, ..., 0.0901674 , 0.4323442 ,
        0.23069155],
       [0.94965756, 0.8568523 , 0.06734467, ..., 0.606243  , 0.04353523,
        0.79850745]], dtype=float32)>
>
```

8. using linux systems in virtual machines to run deep learning scripts

`scp   -r   root@139.59.242.148:/home/phil-jurisprudence-recsys/   D:/projects`: copies a folder and its files from our remote server to our local machine, note we must do this in our local machine as we did copying a file from our lcoal machine to the remote machine

`bash <script-name-here>.sh`: runs the "executable" we downloaded

because the equivalent of .exe files are .sh or shell files in linux we need to run our newly downloaded shell file using bash

sudo apt install python3-pip

sudo apt-get install python3-pip

wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh: downloads an in this case installer file in linux systems

`ssh-keygen`: to generate ssh key that will be placed in `C:\Users\<username>/.ssh/` directory

`more "C:\Users\<username>/.ssh/id_rsa.pub"` will return the generated SSH key using the command we used prior which was `ssh-keygen`

`ssh -i C:\Users\<username>/.ssh/id_rsa <coare username>@<saliksik.asti.dost.gov.ph or 202.90.149.55>`: accesses a remote machine using given IP address

`ls` to list file in current directory

`pwd` to return current or presetn working directory

`scp -r -i "C:\Users\<username>/.ssh/id_rsa" C:\Users\<username>/Documents/Scripts/thesis-writing-1/ <coare username>@saliksik.asti.dost.gov.ph:/home/<coare username>`: copies a folder and its file from our local machine to our remote server

`scp -r -i C:/Users/LARRY/.ssh/id_rsa "C:/Users/LARRY/Documents/Scripts/thesis-writing-1/server-side/modelling/data/Artifact Detection Data/" michael.cueva@202.90.149.55:/home/michael.cueva/scratch1/thesis-writing-1/server-side/modelling/data/`

`scp -i "C:\Users\<username>/.ssh/id_rsa" C:\Users\<username>/Documents/Scripts/thesis-writing-1/<some file>.<ext> <coare username>@saliksik.asti.dost.gov.ph:/home/<coare username>`: copies a file from a local machine to a remote server

`scp -r -i C:/Users/LARRY/.ssh/id_rsa "C:/Users/LARRY/Documents/Scripts/thesis-writing-1/server-side/modelling/data/Artifact Detection Data/test/" michael.cueva@202.90.149.55:'"/home/michael.cueva/scratch1/thesis-writing-1/server-side/modelling/data/Artifact Detection Data/"'`

`scp -r -i C:/Users/LARRY/.ssh/<grup member>/id_rsa "C:/Users/LARRY/Documents/Scripts/thesis-writing-1/server-side/modelling/data/" <group member coare userr name>@202.90.149.55:/home/<group member coare userr name>/scratch1/thesis-writing-1/server-side/modelling/`

`scp -r -i C:/Users/LARRY/.ssh/id_rsa michael.cueva@202.90.149.55:/home/michael.cueva/scratch1/thesis-writing-1/server-side/modelling/saved/ C:/Users/LARRY/documents/scripts/thesis-writing-1/server-side/`: from remote to local

9. calculating L2 norm or eculidean distance in the gaussian kernel of SVM

```
>>> from sklearn.cluster import KMeans
>>> kmeans = KMeans(n_clusters=10).fit(last_hidden_state)
>>> centers = kmeans.cluster_centers_
>>> centers.shape
(10, 30)
>>> last_hidden_state.shape
(100, 30)
>>> from scipy.spatial.distance import cdist
>>>
>>> L2 = cdist(last_hidden_state, centers, 'sqeuclidean')
>>> L2.shape
(100, 10)
>>>
```

Y = cdist(XA, XB, 'sqeuclidean')
Computes the squared Euclidean distance 
 between the vectors.

```
 >>> test = last_hidden_state[0] - centers
>>> test.shape
(10, 30)
>>> L2_0 = np.dot(test, test.T)
>>> L2_0.shape
(10, 10)
>>>
>>> test = last_hidden_state[0] - centers[0]
>>> test.shape
(30,)
>>>
>>> np.dot(test, test.T)
83.23545875066529
>>>
>>> test = last_hidden_state[0] - centers
>>> test.shape
(10, 30)
>>>
>>> powed = tf.pow(last_hidden_state[0] - centers, 2)
>>> powed.shape
TensorShape([10, 30])
>>>
>>> summed = tf.reduce_sum(powed, axis=1)
>>> summed
<tf.Tensor: shape=(10,), dtype=float64, numpy=
array([83.23545875, 55.29155317, 46.37284006, 46.90723522, 40.29037022,
       47.98165298, 50.62750442, 23.73518948, 45.56798788, 41.73636237])>
>>>
```



```
>>> import numpy as np
>>> from scipy.spatial.distance import cdist
>>>
>>> last_hidden_state = np.random.randn(100, 30)
>>>
>>> from sklearn.cluster import KMeans
>>> kmeans = KMeans(n_clusters=10).fit(last_hidden_state)
>>> centers = kmeans.cluster_centers_
>>>
>>> last_hidden_state.shape
(100, 30)
>>> centers.shape
(10, 30)
>>>
>>> tf.keras.backend.expand_dims(last_hidden_state)
<tf.Tensor: shape=(100, 30, 1), dtype=float64, numpy=
array([[[-1.75303964],
        [ 2.5047047 ],
        [-0.89850994],
        ...,
        [-0.03285514],
        [ 1.38745949],
        [-0.24509281]],

       ...,

       [[-1.91323248],
        [-1.38534454],
        [ 2.41042998],
        ...,
        [ 0.10658225],
        [ 1.35406672],
        [ 1.56818429]]])>
>>> tf.keras.backend.expand_dims(last_hidden_state).shape
TensorShape([100, 30, 1])
>>>
>>> expanded = tf.keras.backend.expand_dims(last_hidden_state)
>>> centers.shape
(10, 30)
>>> expanded - centers
>>>
>>> centers.T
array([[ 3.32906516e-01, -5.33232141e-01, -2.50675300e-01,
         1.32915331e-01,  2.30616772e-01,  1.95624663e-01,
        -2.81778413e-01, -7.57086518e-01, -2.01930790e+00,
        -1.70911554e-01],
       ...
       [ 1.92907297e-01, -4.88383123e-03, -6.34055455e-01,
         1.17770622e+00,  2.29998204e-01,  6.05805559e-02,
        -2.27157015e-02,  6.70179414e-01, -1.50471197e+00,
        -6.96626229e-01]])
>>> centers = centers.T
>>> expanded.shape
TensorShape([100, 30, 1])
>>> centers.shape
(30, 10)
>>>
>>> expanded - centers
<tf.Tensor: shape=(100, 30, 10), dtype=float64, numpy=
array([[[-2.08594616, -1.2198075 , -1.50236434, ..., -0.99595313,
          0.26626825, -1.58212809],
        [ 2.20740728,  1.98875262,  2.32314931, ...,  2.99600385,
          2.76492131,  3.23844101],
        [-3.17990036, -0.38634665, -0.12613761, ..., -1.73983374,
         -1.27928614, -0.83260311],
        ...,
        [ 0.57388846,  0.06888886,  0.20212657, ...,  0.2296285 ,
         -3.22137043,  0.29762409],
        [ 0.92152776,  0.7104352 ,  1.17421989, ...,  1.457812  ,
          1.77019652,  2.20857991],
        [-0.43800011, -0.24020898,  0.38896264, ..., -0.91527223,
          1.25961916,  0.45153342]],
       ...,

       [[-2.24613899, -1.38000033, -1.66255718, ..., -1.15614596,
          0.10607542, -1.74232092],
        [-1.68264196, -1.90129662, -1.56689993, ..., -0.89404539,
         -1.12512793, -0.65160823],
        [ 0.12903956,  2.92259327,  3.18280231, ...,  1.56910618,
          2.02965378,  2.47633681],
        ...,
        [ 0.71332585,  0.20832625,  0.34156396, ...,  0.36906589,
         -3.08193304,  0.43706148],
        [ 0.88813499,  0.67704243,  1.14082712, ...,  1.42441923,
          1.73680375,  2.17518714],
        [ 1.37527699,  1.57306812,  2.20223975, ...,  0.89800488,
          3.07289626,  2.26481052]]])>
>>> (expanded - centers).shape
TensorShape([100, 30, 10])
>>>
>>> expanded = tf.expand_dims(last_hidden_state, axis=2)
>>> expanded.shape
TensorShape([100, 30, 1])
>>> centers.shape
(30, 10)
>>> (expanded-centers).shape
TensorShape([100, 30, 10])
>>>
>>> diff = expanded - centers
>>> L2 = tf.reduce_sum(tf.pow(diff, 2), axis=1)
>>> L2.shape
TensorShape([100, 10])
```

10. we can actually run python scripts even from different locations and directories i.e. if we are in a directory `(thesis-writing-1) D:\Projects>` and the script is relative to this directory we can run in by enclosing the path to the script by quotation marks `"<relative or absolute path to script>.<extension>"`. Running `python "./to github/thesis-writing-1/test.py"` for instance will output `testing`

11. you can run vs code via command `code .`

12. Now I know when copying a repository to another machine and then enterign the remote origin url and then adding, committing, and pushing to it requires your github username and password, the password entered can be based from the github classic personal access token you must create in order to push to push to such a remote repository from a remote or another machine besides your local

13. in order to run a `.sh` file in command prompt you need to add `C:\Program Files\Git\bin` as a value to the PATH environment variable. That way we don't need to create anymore `.bat` files that only windows can recognize but not linux, as `.sh` files can be run via `sh <name of shell file>.sh` and `bash <name of shell file>.sh` in windows and linux os' respectively

14. https://dmitripavlutin.com/react-useeffect-explanation/#21-dependencies-argument:
```
useEffect(() => {
    // Runs once, after mounting
    document.title = 'Greetings page';
  }, []);
```

is akin to `componentDidMount()` and 

```
useEffect(() => {
    // Side-effect uses `prop` and `state`
}, [prop, state]);
```

is akin to `componentDidUpdate()`

15. delete diretory in linux without constant prompting: `rm -rf <directory name>`
16. moves directory to another directory: `mv <my folder> /home/<coare username>/...`
17. 
```
Your output from model.predict_proba() is a matrix with 2 columns, one for each class. To calculate roc, you need to provide the probability of the positive class:

Using an example dataset:

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
rf = RandomForestClassifier()
model = rf.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
It looks like this:

array([[0.69, 0.31],
       [0.13, 0.87],
       [0.94, 0.06],
       [0.94, 0.06],
       [0.07, 0.93]])
Then do:

roc_auc_score(y_test, y_proba[:,1])
```
18. `sed -i 's/\r$//' <filename_of_script>`: to convert slurm script to a unix char standard file
19. those long lines of epochs I see in the slurm script output are the DL tnsorflow models being used for prediction on the training set and cross validation set 
20. `numpy.core._exceptions._ArrayMemoryError: Unable to allocate 71.9 GiB for an array with shape (12277, 786318) and data type float64` especially if MinMaxScaler() or scaling is not applied on the data before it is passed in a numpy array
21. a knot in creating splines to smooth data is this -----o------o------o----- where our "lifeline" has multiple knots tied in it in order to use perhaps as gauge in measuring something or how deep can a ships anchor can go.

- A Spline is essentially a piecewise regression line
- You can tailor your line to fit one area wel
-Instead, we break up the observation into different “knots” and fit a different regression line on each segment divvied up by these knots or division points.
- looking at the input for a spline, we are using a 1D list or array.
heart rate for one patient and recorded it every second for 30 seconds. This creates 1 observation with 30 features, or the heart rate recorded at each of those 30 seconds
- We will then fit a Spline to this one observation detailing our knots (t) and order (k), which will return a line of best fit that has its own coefficients
- If our goal was simply to smooth the data and remove noise, we could:
- Fit/train the spline on the observation
- Extract the heart rate by feeding in the exact times we would like to predict the heart rate for (in this example, a list/array of range 0:30) Save this output as the new data for each of our 30 features
- First, this is our function to evenly distribute the locations of our knots (and account for buffer knots depending on the degree chosen) as we go to set the basis for our splines.
```
def knot_points(n_knots, x, degree):
    # create the knot locations
    knots = np.linspace(x[0], x[-1], n_knots) 
    
    lo = min(x[0], knots[0]) #we have to add these min and values to   conform by adding preceding and proceeding values
    hi = max(x[-1], knots[-1])
    augmented_knots = np.append(np.append([lo] * degree, knots), [hi] * degree)
    return augmented_knots

loo = LeaveOneOut()
```  

here we base our knots based on the maximum and minimum value in our dataset or in this case our signals. And generate some number of random values in between this minimum and maximum values based on the number of knots or ties we have in our "lifeline" 
22. important article of SVM implementation in tensorflow 
https://github.com/keras-team/keras/issues/2588
https://github.com/keras-team/keras/issues/12960
https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction

basically what they're trying to say here is in a single script wee ought to train a tensorflow model as a side task and as a byproduct of its training we extract an intermediate layer for its features and use it as input to the second phase of training in the script which will be the feeding of these layer features as input to the SVM 

23. saving output of lstm feature extractor as .npy file     
https://note.nkmk.me/en/python-numpy-load-save-savez-npy-npz/#save-an-array-to-an-npy-file-npsave

24. It sounds like you're encountering an issue with a custom TensorFlow model where the inputs and outputs attributes are not available. This can happen when you create a model using subclassing instead of the functional API. In subclassed models, TensorFlow does not automatically track the inputs and outputs attributes.

25. sometimes outputs of jupyter notebooks can be so big that committing these changes may result in file being too large error and require you to use git large, or pulling these changes may end up in remote hang error

26. The choice of F1-score variant (weighted, micro, or macro) depends on your specific use case and the distribution of your classes. Here's a brief explanation of each:

* Weighted F1-score:
Calculates the F1-score for each class and then computes a weighted average based on the support (number of samples) of each class.
This is useful when you have imbalanced classes and want to give more importance to the minority classes.

* Micro F1-score:
Calculates the F1-score based on the overall number of true positives, false positives, and false negatives across all classes.
This is useful when you want to evaluate the overall performance of the model without considering the class distribution.

* Macro F1-score:
Calculates the F1-score for each class and then computes the unweighted average.
This is useful when you want to evaluate the performance of the model for each class equally, regardless of the class distribution.
Choosing the right F1-score:

If your classes are imbalanced, use the weighted F1-score to give more importance to the minority classes.
If you want to evaluate the overall performance of the model without considering class distribution, use the micro F1-score.
If you want to evaluate the performance of the model for each class equally, use the macro F1-score.
You can also experiment with different F1-score variants to see which one best suits your specific needs.

27. One can also use filezilla as a GUI instead of a command line interface for accessing the DOST COARE HPC resource or virtual machine

* Open FileZilla and navigate to the File menu, select Site ManagerFrom My Sites, select the site you're updating or otherwise create a New SiteSet the following values in the settings:
* Protocol: SFTP - SSH File Transfer Protocol
* Host: Your server hostname or IP address i.e. 202.90.149.55 
* Port: 22 (however coare dost asti does not have a port so leave this blank)
* Logon Type: Key file
* User: Your SFTP username i.e. michael.cueva or whatever username coare verified in your application for an account
* Key file: Use Browse to select your SSH private key or otherwise add the path to your SSH private key. For MacOS users, your private SSH key is typically stored at /Users/USERNAME/.ssh/id_rsa (replace USERNAME with your device username).


## artifact detection and correction:
* <s>clone and review repo of Taylor et al.</s>
* <s>how to segment signals into 0.5s and 5s epochs/windows</s>
* <s>decompose signal into phasic and tonic components</s>
* detect and correct the artifacts using LSTM-SVM and Llanes-Jurado et al. correction pipeline
* find dataset about stress detection with raw eda signals
* <s>find out what BioVid and PainMoinit dataset from gouverneur et al. (2023) to trace the code they wrote</s>
* <s>review repo of gouverneur et al. (2023) since it details how autoregressive feature extraction,</s>
* <s>research how VFCDM from Hossain et al. (2022) works</s>
* review repo of gouverneur et al. (2023) to learn how cvxEDA works

* <s>review repo of gouverneur et al. (2023) to learn how RFE-LOSO 5-fold cross validation works
1. There are 43 subjects, 33 of which will be used for training & validation and 10 of which will be used for testing 
2. All 43 subjects signals will undergo extensive feature extraction prior by running feature_engineering.ipynb
3. The extracted features from each corresponding subjects recorded signals will now be really split into 33 and 10, the other for trianing and the other for testing
4. These 33 & 10 subject features will be placed inside a training folder & a testing folder
5. We then load the newly engineered train data
6. Data is then used for feature selection by using recursive feature elimination to select best features
6. A grid search loop is then setup using these best features such that all possible hyper param configurations are used
7. in each iteration LOSO cross validation will be carried out
    ```
    for hyper param 1 in hyperparams 1
        for hyper param 2 in hyperparams 2
            ...
            for hyper param n in hyperparams n
                loso_cross_validation(features/X, y/labels, subjects, model)
    ```

8. loso_cross_validation() will
    - use best features for all 33 subjects
    - run a loop tthat will leave one subject out where for all subjects...
        - these will be our folds
        - for each iteration of this loop will a classifier is trained
        - the classifiers score is then recorded at this fold & move on to next iteration
    - average out all scores collected at each "fold", associate it with the best selected features and the hyper param configuration it used
9. save the best features found by the RFE algorithm</s>

* <s>visualize results of each model with specific hyper param config at each fold i.e. 
at fold 1 this is the cross and train accuracy, with hyper param config 1 
at fold 2 this is the cross and train accuracy, with hyper param config 1
at fold n this is the cross and train accuracy, with hyper param config 1
x-axis could be the folds, y-axis could be the cross and train accuracy and other metrics</s>

* <s>visualize overall mean train and mean cross accuracy and other metrics across all hyper params i.e.
mean train accuracy and mean cross accuracy with hyper param config 1
mean train accuracy and mean cross accuracy with hyper param config 2
mean train accuracy and mean cross accuracy with hyper param config n
x-axis could be hyper param a in hyper param config 1, hyper param a in hyper param config 2</s>

* <s>implement different loading strategy for llanes-jurado et al. (2023) pipeline which would be loading the raw eda signals and then converting it to 3d matrices</s>

* <s>I will then have to find some way to assign subject ids to these 3d matrices so that it can be compatible with loso cross validation, either way I will have to finish the functions that will be loading data for Llanes-Jurado model</s>

* <s>implement loso cross validation using llanes-jurado model</s>
what you can do if you want to implement LOSO validation here is to leave one subject out per subject:
1. feed to the model the training data
model.fit(training subjects signals)
2. once trained allow the model to predict the validation data or the left out subject
model.predict(validation subject signals)
3. extract the predicted labels and calculate different performance metric values i.e. bce_loss, Accuracy, precision, recall, f1-score, auc
bce_loss(Y_true, Y_pred), Accuracy(Y_true, Y_pred), Precision(Y_true, Y_pred), ...
record the resulting 

the most important result here is not the specific hyper param configs train and validation across 100 epochs results

the most importatnt result here is the hyper param configs final train and validation in a fold or a set subjects signals

* <s>do final checks for LSTM-SVM and LSTM-CNN in terms of saving their respective metric values for each hyper param configs. Do final checks also of LSTM-SVM since I still doubt its effectiveness given it doesn't have the C parameter in its loss function</s>
https://github.com/AryaAftab/svm-tensorflow/tree/master/svm_tensorflow

* <s>once models are tuned the results will be collected and saved in a .json file
from here we can suse summarization.ipynb and review which models perform the best, once we've
decided what models with certain hyper params work best we then train these models fully and save
them</s>

* <s>we then use the saved models to predict the final test data</s>
* we use the predicted data and use it as basis to correct the eda signals
* <s>start implementing the LSTM-SVM model</s>
* <s>test llanes-jurado model on notebook and see if dataset will work</s>
* review how automatic_EDA_correction() function by llanes-jurado works in order to start using the predicted data as basis to correct the eda signals

* <s>pqbqpr_expert2.csv</s>
calculating features from 1970-01-01 00:00:00 to 1970-01-01 00:00:00.500000 for index 0
calculating features from 0 to 63 for index 0
#### it takes 63 numbers excluding 0 to get from 0 to 63
calculating features from 1970-01-01 00:59:59 to 1970-01-01 00:59:59.500000 for index 7198
calculating features from 460672 to 460735 for index 7198
#### 672,673,674,675,676,677,...,733,734,735 it takes 63 numbers excluding 460672 to get from 460672 to 460735

460736 + 63 = 460798
in theory we could still take indeces 460736 to 460736 + 63 which is 460799 which is supposed to be exactly the end index of the hour long data slice 

#### so for index 7199 which will be zeroes 

#### only strictly until before an hour

##### this is our hour long data slices indeces exclusively, i.e. [0] until [460799] not including 460800 since this would be in the 1 hour mark and would be longer than an hour
processed hour 0 - start: 0 | end: 460800

calculating features from 1970-01-01 01:00:00 to 1970-01-01 01:00:00.500000 for index 0
calculating features from 460800 to 460863 for index 0
calculating features from 1970-01-01 01:59:59 to 1970-01-01 01:59:59.500000 for index 7198
calculating features from 921472 to 921535 for index 7198
processed hour 1 - start: 460800 | end: 921600

calculating features from 1970-01-01 02:00:00 to 1970-01-01 02:00:00.500000 for index 0
calculating features from 921600 to 921663 for index 0
calculating features from 1970-01-01 02:07:36.500000 to 1970-01-01 02:07:37 for index 913
calculating features from 980032 to 980095 for index 913
processed hour 2 - start: 921600 | end: 980118



oxused_expert2.csv
hour long timestamp list length: 7200
calculating features from 1970-01-01 00:00:00 to 1970-01-01 00:00:00.500000 exclusively for index 0
calculating features from 0 to 63 inclusively for index 0

calculating features from 1970-01-01 00:59:58.500000 to 1970-01-01 00:59:59 exclusively for index 7197
calculating features from 460608 to 460671 inclusively for index 7197

calculating features from 1970-01-01 00:59:59 to 1970-01-01 00:59:59.500000 exclusively for index 7198
calculating features from 460672 to 460735 inclusively for index 7198

last timestamp: 1970-01-01 00:59:59.500000
last timestamp of exclusion timestamp list for first 1 hour slice: DatetimeIndex([   '1970-01-01 00:59:59.500000',
...
'1970-01-01 00:59:59.992187500'], or 

0:59:59.96875
0:59:59.9765625



last timestamp of exclusion timestamp list for second 1 hour slice:
 '1970-01-01 01:39:36.906250' or '5976.90625' and when we do check oxused_expert2.csv's last row which has this timestamp when 

* <s>calculating roc auc score from predicted and true labels</s>

```
def get_auc_score(y_true, y_prob):
    y_true = y_true.values
    y_prob = y_prob.values
    
    # Get indices that would sort y_prob in descending order
    sorted_indices = np.argsort(-y_prob)
    
    # Sort y_true using the same indices
    sorted_y_true = y_true[sorted_indices]
    
    # Compute total number of positive and negative samples
    num_positive = np.sum(sorted_y_true == 1)
    num_negative = np.sum(sorted_y_true == 0)
    
    # Initialize arrays for TPR and FPR
    tpr_array = []
    fpr_array = []
    
    cum_tp = np.cumsum(sorted_y_true == 1)
    cum_fp = np.cumsum(sorted_y_true != 1)
    
    tpr = cum_tp / num_positive
    fpr = cum_fp / num_negative

    tpr_array.append(tpr)
    fpr_array.append(fpr)
        
    return np.trapz(tpr_array, fpr_array)
```

* <s>there still needs to be a way to prevent data leakage in loading data for the DL models because a sort of normalization function,</s>
- I pass validation data percentage of 0.3 to the training of the model
- so new pipelien would be to read the read the signals as usual without the min max scaler anymore
- separately join the training signals, and the cross validation signals
- min max scale training signals, transform cross validation signals based on min max scaler scaled on training signals
- save min max scaler scaled ontraining signals for later use in evaluation
**this is closed as the only reason why min max scaling is applied for the signals is not because normalization is needed but because loading data by itself without scaling will take up too much ram for the machine to tolerate**

* <s>as far as I know I have the ff. models to tune and train
- hossain_lr
    - done with tuning and training
- hossain_gbt
    - will push through with tuning but with RFC
- hossain_svm
    - will not push through with tuning anymore, will go straight to training
    - train with gamma and C parameters 0.5 and 0.1
- taylor_lr
    - will push through with tuning but with RFC
- taylor_rf
    - will push through with tuning but with RFC
- taylor_svm
    - will not push through with tuning anymore, will go straight to training
    - train with gamma and C parameters 0.5 and 0.1
- jurado_lstm_cnn
    - already have weights that have been tuned and trained 
- cueva_lstm_svm
    - need lstm feature extractor trained
    - need svm connection trained also on feature extractor will not tune anymore and go straight into training
    - gamma and C params 0.5 and 0.1

and at the end of this once models are trained they will be evaluated on the testing data
and whatever scores the trained models get will be recorded, hindi na yung galing sa hyper param
tuning or summarization.ipynb</s>

* <s>get stress detection data
- decompose eda signals in stress detection data to phasic and tonic component
- train dl or ml model on phasic component of the raw eda signals
- normallize or not normalize?? If we decide to normalize train data</s>

## SVM mechanism
* <s>I need to learn how SVM can be implemented in tensorflow</s>
* <s>I need to learn also how SVM works explained by Andrew Ng</s>

## stress detection:
* <s>beacuse stress detection requires raw eda signals be corrected deconvolved into phasic components we will have to find a dataset that has raw eda signals labeled with stress or non stress and train a second model</s>
* <s>once the EDABE dataset has been denoised or the artifacts detected and subsequently corrected, we will have to use this second model trained on stress detection data to identify indeed what segments of our denoised EDABE dataset indicate a heightened level of stress</s>

## Recording problems after copying of:
* <s>`pip install -r requirements.txt` gives version error even if python version is supposed to be compatble with the package versions</s>
* <s>`conda create -n thesis-writing-1 python=3.12.3` sometimes gives error</s>
* <s>vs code not being able to download server with host 202.90.149.55 or saliksik.asti.dost.gov.ph</s>
* <s>extremely slow installation of packages</s>
* <s>unable to `git push origin master` always requires authentication and when I do input my credentials it throws `fatal authentication error` but weirdly able to `git add .` and `git commit -m "update"`. I've tried adding my username and email via `git config --user.name "<my github user name>"` and git config --user.email "<my email>"</s>

## Running scripts in HPC:
* <s>if we were to use a gpu our partition and qos parameters would be set to the following valuess
```
#SBATCH --partition=gpu
#SBATCH --qos=gpu-p40_default
#SBATCH --gres=gpu:1  ## or --gres=gpu:p40:1
```

```
#!/bin/bash
#SBATCH --account=<slurm_group_acct>
#SBATCH --partition=batch
#SBATCH --qos=batch_default
#SBATCH --nodes=36
#SBATCH --ntasks=86
#SBATCH --job-name="<jobname>"
#SBATCH --output="%x.out"         ## <jobname>.<jobid>.out
##SBATCH --mail-type=ALL          ## optional
##SBATCH --mail-user=<email_add>  ## optional
##SBATCH --requeue                ## optional
##SBATCH --ntasks-per-node=1      ## optional
##SBATCH --mem=24G                ## optional: mem per node
##SBATCH --error="%x.%j.err"      ## optional; better to use --output only

## For more `sbatch` options, use `man sbatch` in the HPC, or go to https://slurm.schedmd.com/sbatch.html.

## Set stack size to unlimited.
ulimit -s unlimited

## Benchmarking.
start_time=$(date +%s.%N)

## Print job parameters.
echo "Submitted on $(date)"
echo "JOB PARAMETERS"
echo "SLURM_JOB_ID          : ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME        : ${SLURM_JOB_NAME}"
echo "SLURM_JOB_NUM_NODES   : ${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_NODELIST    : ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS          : ${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE : ${SLURM_NTASKS_PER_NODE}"
echo "SLURM_MEM_PER_NODE    : ${SLURM_MEM_PER_NODE}"

## Create a unique temporary folder in the node. Using a local temporary folder usually results in faster read/write for temporary files.
custom_tmpdir="yes"

if [[ $custom_tmpdir == "yes" ]]; then
   JOB_TMPDIR=/tmp/${USER}/SLURM_JOB_ID/${SLURM_JOB_ID}
   mkdir -p ${JOB_TMPDIR}
   export TMPDIR=${JOB_TMPDIR}
   echo "TMPDIR                : $TMPDIR"
fi

## Reset modules.
module purge
module load <module1> [<module2> ...]

## Main job. Run your codes and executables here; `srun` is optional.
[srun] /path/to/exe1 <arg1> ...
[srun] /path/to/exe2 <arg2> ...

## Flush the TMPDIR.
if [[ $custom_tmp == "yes" ]]; then
   rm -rf $TMPDIR
   echo "Cleared the TMPDIR (${TMPDIR})"
fi

## Benchmarking
end_time=$(date +%s.%N)
echo "Finished on $(date)"
run_time=$(python -c "print($end_time - $start_time)")
echo "Total runtime (sec): ${run_time}"
```
</s>

* <s>currently running scripts by HPC are `hossain_lr_tuning_job.sbatch`, `hossain_svm_tuning_job.sbatch`, and `hossain_gbt_tuning_job.sbatch`</s>
* <s>so far by products have only been `reduced_hossain_lr_feature_set.txt` which tells me that the `hossain_lr_tuning_job.sbtach` is running fine albeit still in the tuning process</s>



## Front end setup
* <s>install node.js</s>
* <s>run `npm create vite@latest client-side -- --template react` or `npm create vite@latest` and just follow the subsequent prompts that will ask you what template you want to use i.e. react, svelte, vue etc. and what name will the project be, in directory `~/thesis-writing-1`</s>
* <s>start developing in react so its quick</s>

## Building components
* <s>setup css files first by copying from the `project-seraphim` project template in order to build navbar</s>
* <s>build navbar</s>
* <s>search some way how to visualize time series data like eda signals in react</s>
* <s>search some way to upload .csv files to backend via frontend my guess is this would be akin to submitting an image like file like what you did in the micro organism classifier</s>
* need some way to show and visualize artifacts in uploaded signal
* need some way to show and visualize corrected parts of the uploaded signal
* need some way to show and visualize which parts of the signal are detected as stressful events using the `stress_detector.pkl` model
* design faq section
* design and implement researchers section
* edit svg logo of application in photoshop
* edit svg logo of dost, coare, and dost-asti to be black and white in photoshop for footer component of website


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

# JUST NEED TO:
* <s>send emails to gc and ask them to send their respective emails</s>
* <s>follow up the group to send their emails (pukpukin mo if you have to)</s>
* <s>ask taline for paper review or if not I will alone</s>
* <s>send emails to every single one of those researchers</s>
* <s>prepare for possible zoom/google meet with them by preparin jupyter notebook, asking about uneven length wavelet features,</s>
* <s>send email to hossain et al. </s>
* <s>fill up and sign agreement form to access BioVid database data and send to sascha.gruss@uni-ulm.de</s>

* <s>send out aplication to COAREs for computing services</s>
1. +63 (02) 8426-3572, gridops@asti.dost_gov.ph, coareservice@asti.dost.gov.ph
2. you'll need an account for COARE
3. apply for account to access COARE services like high perf. computing for training models on large data and science cloud for cloud based applications
4. to submit a request write first email to the above by providing brief overview of research
5. download and fill up endorsement letter basically stating why you want to use COARE's computing services and why your research/thesis/dissertation needs it and have your endorser be as much as possible a faculty of PUP or your institution
6. accomplish COARE application
7. you the applicant and DOST-ASTI must agree on an agreement
8. after COARE application submission, the COARE team will eval it 
    - if approved team will give your COARE account credentials
    - if not team will ask you to revise or adjust the applicatoin and maybe ask you for additional requirements. Revised app. will be reevaled.
9. after approval of app team will create your COARE acc
10. team will email your acc. credentials and instructions for first time users

* <s>we already have our request approved next steps now is to use the ssh key and enter the VM with the 250gb ram and 100gb storage and we can get to tuning and then training the models. But before this I have to do a cleanout of the test models because we will now be using the space for saving the real tuned and trained models</s>

* Solve the following critical questions from Maam Alet
- Kahit isang previous model lang ang maungusan ni LSTM-SVM okay na, basta kailangan mas mataas yung important metrics niya like f1, prec, recall, and roc-auc at least kaysa sa isang previous
- Bawal mas mababa yung metrics ng lstm svm sa existing 
- Or what is the reason bakit mas mababa yung model ngayon kasi nag experiment 
- Nakakatakot pag walang significant difference. 
a. Pero what is somehow good is that test accuracy of LSTM-SVM is 2% higher than both logistic regression and lstm-cnn
b. test f1 of LSTM-SVM is 2% higher than logistic regression, 1% higher than svm, and 0.2% higher than lstm-cnn 
c. test precision of lstm-SVM is 0.1% higher than logistic regression, but 2% lower than other models
c. test recall of lstm-svm is 2% higher than logistic regression, 2% higher than lstm-cnn
d. test roc-auc of LSTM-SVM is higher than hossain et al. (2022) svm by 2% while being lower than all others and being lower by 11% than the lstm-cnn 

the 1-2-weighted-svm might probably be the best one since it outperforms the lstm-cnn in terms of test accuracy, outperforms taylor et al. (2015) lr and svm in terms of test f1-score, outperforms taylor et al. (2015) lr in terms of precision, outperforms taylor et al. (2015) and hossain et al. (2022) lr and svm by a whopping 4%, 3%, 1%, and 8% respectively, lstm-svm also outperforms lstm-cnn by 2% in terms of true negative rate or specificity, lstm-svm outperforms all other models in terms of true positive rate or sensitivity except lstm-cnn which is still ahead by 11% 


- If imbalanced ang dataset, bakit mas mataas sa iba? bakit ang proposed model ay di nakakaungos sa iba? Anong problem? This is a big one because how caan we know the answers? Do we scour the internet to find the answer? How do our experiments reveal this? Kaya we must never saay na imbalanced ang dataset kasi pupuntiryahin tayo. Ok so solutions can be to tackle this are the ff.
a. oversampling data of the minority class by introducing synthetic datapoints
b. undersampling data of hte majority class by deleting some of the datapoints in the majority class, because there may be rows that include repetitions (After trial and error it increased the score by 8%)
* https://datascience.stackexchange.com/questions/106642/improving-performance-of-machine-learning-for-a-small-imbalanced-dataset
* https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
c. try class weighted svm, because there is an imbalance of data with ratio of 80 to 20 on the negative to positive samples, a weighted svm may do the trick by
* <s>https://scikit-learn.org/dev/auto_examples/svm/plot_separating_hyperplane_unbalanced.html</s>
* <s>wj=n_samples / (n_classes * n_samplesj) e.g. if we have 400000 rows, 2 unique classes, with the negative class having 360000 rows, and the positive class having only 40000 rows thus havving a ratio of 9 to 1, then the weights we calculate for both classes will be 400000 / (2 * 360000) = 0.55 and 400000 / (2 * 40000) = 5 for the 0 and 1 labels respectively, this would in sklearn look like `class_weight={0:0.55,1:5}`. we can also do this by using the `class_weight` object from `sklearn.utils` i.e.</s>

```
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
```

* <s>we already tried a class weight of 1 to 8.86 for all models including lstm-svm they show promising increase in test roc-auc however there tends to be a slight over prediction of 1s over 0s maybe try lessening the weights of the positive class 1 to 6.5 such that our argument value for `class_weight` would now be `{0: 1, 1: 6.5}`. We also already tried using a balanced class weight of 0.5 to 4.9 however this ratio when divided by 0.5 still comes out to 1 to 9.8 which is even higher than 8.86. so we need to redo training again and try a ratio of 1 to 6.5 to lessen the over prediction of models of 1s</s>

* <s>tried a ratio of 1 to 5 not 1 to 6.5 but the former and it still overpredicts ones and even lowered the precision, recall, and f1-score previously</s>

* I can try 1 to 2.5 and lower it some more, if 2.5 performs the best then it could replace the 1 to 9 ratio (or 0.5 to 4.9) but if not then we keep the latter ratio and label it the best model

* change the feature selector to have no class weights as the features it could choose could have an effect on the svm model itself due to it being weighted so redo feature selection

* <s>another thing I can do is retrain the lstm model on the imbalanced dataset and set the `model.fit()`s `class_weight` arg to the calculated class weights and again use the model to predict. To expensive and did it to no avail in the end</s>

d. removed collinear features
e. use polynomial or linear kernel
f. try to visualize the features of the data to see what kind of kernel may be best suited for the data
g. try to retrain a feature extractor LSTM again but with more powerful hyperparams

because...
a. I've already tried removing outliers in the training dataset

- Yung unang picture ng system architecture ay nasa work flow part nalang 

- If supervised, dapat labeled yung data. Meron naman labels ang data i.e. Artifact and Non-Artifact

- Transition (Tool) para maintindihan ng non-IT people, meaning the app must be able to smooth out the signals with noise by removing the noise via detecting the parts with noise and then smoothing out these parts

* https://stackoverflow.com/questions/72581298/how-to-zoom-only-on-the-x-axis-in-a-line-chart-in-d3-js-v7
* https://d3-graph-gallery.com/graph/area_brushZoom.html

* PS: the M command of the "d" attribute uses (as pretty much everything in every language) the coordinate pair x,y (horizontal/vertical). Thus, when you have the error:
`MNaN,42...`
...you know the problem is in your x scale, while the error:
`M42,NaN...`
...shows you the problem is in the y scale.

so what does the error `error path attribute d: Expected number, "M-1014.345,NaNL-1014.343,Na…".` mean?


```
import * as d3 from "d3";
import React, { useState, useEffect, RefObject } from "react";

function Chart() {
  const svgRef: RefObject<SVGSVGElement> = React.createRef();

  function toggleFill() {
    if (fill === "red") {
      setFill("blue");
    } else {
      setFill("red");
    }
  }

  const [fill, setFill] = useState("red");

  const [zoomState, setZoomState] = useState(d3.zoomIdentity);

  const width = 500;
  const height = 400;
  const margin = { top: 50, right: 50, bottom: 50, left: 100 };
  const innerHeight = height - margin.top - margin.bottom;

  const data = [
    [-100, 100],
    [-100, 200],
    [-100, 300],
    [-100, 400]
  ];
  useEffect(() => draw(), [fill, zoomState]);
  function draw() {
    let yScale = d3.scaleLinear().domain([100, 400]).range([innerHeight, 0]);

    if (zoomState) {
      const newYscale = zoomState.rescaleY(yScale);
      yScale.domain(newYscale.domain());
    }

    let yAxis: any = d3.axisLeft(yScale);

    const zoomer: any = d3.zoom().on("zoom", zoom);

    function zoom(event: any) {
      const zoomState = event.transform;
      setZoomState(zoomState);
    }

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .call(zoomer);

    svg
      .select("text.button")
      .attr("transform", "translate(200,20)")
      .on("click", toggleFill);

    svg
      .select("#circles-group")
      .attr("transform", "translate(200, 0)")
      .selectAll("circle")
      .data(data)
      .join("circle")
      .style("fill", fill)
      .attr("r", 4)
      .attr("cx", function (d) {
        return d[0];
      })
      .attr("cy", function (d) {
        return yScale(d[1]);
      });

    svg.select("#y-axis").attr("transform", "translate(75,0)").call(yAxis);
  }

  return (
    <div
      style={{ margin: 50, position: "relative" }}
      className="OptionsDataChart"
    >
      <svg ref={svgRef}>
        <text className="button">toggle fill</text>
        <g id="circles-group" />
        <g id="y-axis" />
      </svg>
    </div>
  );
}

export default Chart;
```

- Paano tinatanggal yung error. Easy yung correction pipeline/smoothing function nila Llanes-Jurado et al. (2023) ginamit namin
- Evaluator para masabi kung tama yung na smooth na signals. Easy kasi hindi naman tayo ang gumawa ng smoothing function sila Llanes-Jurado et al. (2023) they set the standard and so we are just merely using what they already had to correct the signals since it is out of our scope to build our own signal correction pipeline. Pero need pa rin ng written approval kina Llanes-Jurado et al. (2023) proving that the correction pipeline we used is theirs and that they can vouch that the signals that were corrected using their correction pipeline are verified and approved
- Saan metric nakikita yung smoother, kilangan rin matanong kina Llanes-Jurado et al. (2023) how they had the justification to say that the corrected signals were suitable for other tasks like stress detection. 

- As the user clicks the user must have the option of seeing the noise and the corrected noise
* https://d3-graph-gallery.com/graph/interactivity_zoom.html
* https://codepen.io/jjjj60110/pen/RMLBpG

- Parameters of SVM 
- Specify SOP2 sabihin niyo nalang "...SVM part of the LSTM-SVM model"

- If you say imbalanced, anong ginawa para di imbalanced si dataset?

- Why use svm in the first place? We wanted to use an svm in place of a cnn because Taylor et al. (2015) used it in their study and we used it as a framework if we could achieve a similar if not a better result to Llanes-Jurado et al. (2023)'s study. We also used SVM as some studies like with Agarap et al. (2022) also proved that traditional ML models (although not SVM as XGBoost was used) achieved comparable results to deep learning architectures

- If the LSTM-SVM you have now only has comparable results to other models what's the point of addressing the gap when the gap itself of supposedly traditional ML models that use only lower order features achieve lower results when compared to DL architectures wasn't solved? Yes although our model doesn't entirely address the currently present problem because of numerous reasons such as class imbalance, choice of model i.e. SVM, this could serve as a benchmark for future researchers to not only balance out the class imbalance present in the data to achieve better results but to use perhaps other traditional ML models instead of an SVM i.e. Gradient Boosted Trees, Xgboost, Random Forest, etc.

- What is really the gap? It was the problem of ML models having lower performance because of low level features as opposed to DL models that were able to extract higher order features. We proposed a solution that could not only use low level features but incorporate higher level features, with respect to previous models

- send follow up email to llanes-jurado


"Larry, Taline, Deseree, and Wana will graduaate in 2025"