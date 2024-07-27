import tensorflow as tf


"""
1. so I have 43 subjects, 33 of which I use for training & validation and 10 of which will be used for testing 
2. All 43 subjects signals will undergo extensive feature extraction by running feature_engineering.ipynb
3. the extracted features from each corresponding subjects recorded signals will be now really split into 33 and 10, the other for trianing and the other for testing
4. these 33 & 10 subject features will be placed inside a training folder & a testing folder
5. the first and foremost step is loading only the newly engineered train data
6. setting up a grid search loop such that all possible hyper param configurationsare used 
7. in each iteration LOSO cross validation will be carried out
    ```
    for hyper param 1 in hyperparams 1
        for hyper param 2 in hyperparams2
            ...
            for hyper param n in hyperparams n
                LOSO_CV(features/X, y/labels, subjects, model)
    ```

8. LOSO_CV will
    - use backward feature selection first and select best features
    - use best features for all 33 subjects
    - run a loop tthat will leave one subject out for all subjects these will
        - these will be our folds
        - for each iteration of this loop will we train our classifier
        - record this classifiers score at this fold & move on to next iteration
    - average out all scores collected at each "fold", associate it with the best selected features and the hyper param configuration it used
"""