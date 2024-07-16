import numpy as np
import pandas as pd

import matplotlib as mplt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
font = {'fontname': 'Helvetica'}

import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sb
import networkx as nx

from sklearn.metrics import (
  accuracy_score, 
  precision_score, 
  recall_score, 
  f1_score, 
  roc_auc_score, 
  mean_squared_error, 
  mean_absolute_error)
from sklearn.manifold import TSNE

import itertools

def view_time_frame(raw_eda_df, samp_freq=128, begin_time_s=1750, end_time_s=1765, cols_to_use=['rawdata', 'cleandata', 'signal_automatic']):
    """
    ntoe cols to use must be equal to 5 or more
    """
    fig = plt.figure(figsize=(17, 5))

    axis = fig.add_subplot()


    # why is it multiplied by 128?
    # results it 224000 and 225920 and is used as indeces to access a slice of the dataframes rows
    begin_sample, end_sample = begin_time_s * samp_freq, end_time_s * samp_freq

    # 
    time_to_plot = raw_eda_df["time"].iloc[begin_sample:end_sample]

    # colors and linestyles to use
    colors = ['#df03fc', '#5203fc', '#fc034e', '#fc8003', '#3dfc03']
    lines = ['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3))]

    for i, col in enumerate(cols_to_use):
        col_to_plot = raw_eda_df[col].iloc[begin_sample:end_sample]
        axis.plot(time_to_plot, col_to_plot, label=col, alpha=0.75, linestyle=lines[i], c=colors[i])
    # axis.plot(time_to_plot, rawdata_to_plot, label="Raw data")
    # axis.plot(time_to_plot, cleandata_to_plot, label="Manual", c="orange")
    # axis.plot(time_to_plot, autodata_to_plot, label="Automatic", alpha=0.7, linestyle="--", c="red")
    # axis.plot(time_to_plot, autodata_to_plot, label="Automatic", alpha=0.7, linestyle="--", c="red")
            
    axis.legend(fontsize=14)
    axis.grid()
            
    axis.set_ylabel(r'$\mu S$', fontsize=16)
    axis.set_xlabel("Time (s)", fontsize=16)

    plt.show()

def view_wavelet_coeffs(coeffs):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 6))
    for i, axis in enumerate(axes):
        axis.plot(coeffs[i], color='r')

        if i == 0:
            axis.set_title("Approximation coefficients", fontsize=14)
        else:
            axis.set_title("Detail coefficients", fontsize=14)
            axis.set_ylabel("Level {}".format(i), fontsize=14, rotation=90)

    plt.tight_layout()
    plt.show()

def analyze(X_trains, feature_names: list, fig_dims: tuple=(4, 2), color: str="#036bfc", img_title: str="untitled", save_img: bool=True, style: str='dark'):
    """
    suitable for all continuous input

    to write:
    dynamic division of features insteawd of 4, 2 what if there are 12 features 
    of the data? 20? 32? 15? What then?

    I can't just write indeces as the title of each subplot, I need to use a list 
    feature names instead since more likely than not a numpy type dataset will be
    used to process visualize these features and the range of their values

    args:
        X_trains - a numpy matrix that will be used to visualize each of its
        individual features and see where each features values range from and to

        feature_names - a list of strings representing the names of each feature
        column, or variable of the dataset/matrix since it is a numpy array in
        which case it would not contain any meta data such as the name of each
        feature, column, or variable
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # see where each feature lies
    # sees the range where each feature lies
    first_dim, zeroeth_dim = fig_dims
    fig, axes = plt.subplots(first_dim, zeroeth_dim, figsize=(15, 10))
    fig.tight_layout(pad=1)

    # no. of instances and features
    num_instances = X_trains.shape[0]
    num_features = X_trains.shape[1]
    
    zeros = np.zeros((num_instances,))
    for feature_col_i, axis in enumerate(axes.flat):
        # extracts the current feature column which will be of 
        # m x 1 dimensionality which we will need to reshape to 
        # just m now
        curr_feature = X_trains[:, feature_col_i].reshape(-1)

        # here we plot the current features array of values
        # to an array of zeroes of m length in order to
        # see the features values only on the x-axis and on the 
        # 0 value only in its y-axis
        axis.scatter(curr_feature, zeros, alpha=0.25, marker='p', c=color)
        axis.set_title(feature_names[feature_col_i], )
        
    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def data_split_metric_values(Y_true, Y_pred, metrics_to_use: list=['accuracy', 'precision', 'recall', 'f1', 'roc-auc'], style: str='dark'):
    """
    args:
        Y_true - a vector of the real Y values of a data split e.g. the 
        training set, validation set, test

        Y_pred - a vector of the predicted Y values of an ML model given 
        a data split e.g. a training set, validation set, test set

        unique_labels - the unique values of the target/real Y output
        values. Note that it is not a good idea to pass the unique labels
        of one data split since it may not contain all unique labels

        given these arguments it creates a bar graph of all the relevant
        metrics in evaluating an ML model e.g. accuracy, precision,
        recall, and f1-score.
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    unique_labels = np.unique(Y_true)

    metrics = {
        'accuracy': accuracy_score(Y_true, Y_pred),
        'rmse': np.sqrt(mean_squared_error(Y_true, Y_pred)),
        'mse': mean_squared_error(Y_true, Y_pred),
        'precision': precision_score(Y_true, Y_pred, labels=unique_labels, average='weighted'),
        'recall': recall_score(Y_true, Y_pred, labels=unique_labels, average='weighted'),
        'f1': f1_score(Y_true, Y_pred, labels=unique_labels, average='weighted'),
        'roc-auc': roc_auc_score(Y_true, Y_pred, labels=unique_labels, average='weighted')
    }

    # create metric_values dictionary
    metric_values = {}
    for index, metric in enumerate(metrics_to_use):
      metric_values[metric] = metrics[metric]

    return metric_values

def view_words(word_vec: dict, word_range: int, title: str="untitled", save_img: bool=True, style: str='dark'):
    """
    suitable for all discrete input

    args:
        word_vec - key value pairs of the words and respective embeddings

        len_to_show - the limit in which each word vector is only allowed to show

        word range - if false then all words are shown but if a value 
        is given then number words shown are up to that value only
        
        word_range: int | bool=50
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # slice the dictionary to a particular range
    sliced_word_vec = dict(itertools.islice(word_vec.items(), word_range))

    # separate all word keys and their respective 
    # embeddings from each other and place in separate arrays
    words, embeddings = zip(*sliced_word_vec.items())
    words = np.array(words)
    embeddings = np.array(embeddings)

    # reduce length/dimensions of embeddings from 300 to 2
    tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)

    # because there are 21624 words dimensionality of emb_red will go from 21624 x 300 to 21624 x 2
    emb_red = tsne_model.fit_transform(embeddings)

    # populate a new dictionary with new reduced embeddings with 2 dimensions
    word_vec_red = {}
    for index, key in enumerate(words):
        # extract x and ys in emb_red array
        x, y = emb_red[index]

        # populate dictionary with x and y coordinates
        if key not in word_vec_red:
            word_vec_red[key] = (x, y)


    # build and visualize
    fig = plt.figure(figsize=(15, 15))
    axis = fig.add_subplot()

    # plot the points
    axis.scatter(emb_red[:, 0], emb_red[:, 1], c=np.random.randn(emb_red.shape[0]), marker='p',alpha=0.75, cmap='magma')

    # annotate the points
    for iter, (word, coord) in enumerate(word_vec_red.items()):
        x, y = coord
        axis.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    axis.set_xlabel('x', )
    axis.set_ylabel('y', )
    axis.set_title(title, )
    
    if save_img:
        plt.savefig(f'./figures & images/{title}.png')
        plt.show()

def view_value_frequency(word_counts, colormap: str="plasma", title: str="untitled", save_img: bool=True, kind: str='barh', limit: int=6, asc: bool=False, style: str='dark'):
    """
    suitable for all discrete input

    plots either a horizontal bar graph to display frequency of words top 'limit' 
    words e.g. top 20 or a pie chart to display the percentages of the top 'limit' 
    words e.g. top 20, specified by the argument kind which can be either
    strings barh or pie

    main args:
        words_counts - is actually a the returned value of the method
        of a pandas series, e.g.
            hom_vocab = pd.Series(flat_hom)
            hom_counts = hom_vocab.value_counts()

        limit - is the number of values to only consider showing in
        the horizontal bar graph and the pie chart

        colormap - can be "viridis" | "crest" also
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # get either last few words or first feww words
    data = word_counts[:limit].sort_values(ascending=asc)

    cmap = cm.get_cmap(colormap)
    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()
    
    if kind == 'barh':        
        axis.barh(data.index, data.values, color=cmap(np.linspace(0, 1, len(data))))
        axis.set_xlabel('frequency', )
        axis.set_ylabel('value', )
        axis.set_title(title, )
        
    elif kind == 'pie':
        axis.pie(data, labels=data.index, autopct='%.2f%%', colors=cmap(np.linspace(0, 1, len(data))))
        axis.axis('equal')
        axis.set_title(title, )
    
    if save_img:
        plt.savefig(f'./figures & images/{title}.png')
        plt.show()

def multi_class_heatmap(conf_matrix, img_title: str="untitled", cmap: str='YlGnBu', save_img: bool=True, style: str='dark'):
    """
    takes in the confusion matrix returned by the confusion_matrix()
    function from sklearn e.g. conf_matrix_train = confusion_matrix(
        Y_true_train, Y_pred_train, labels=np.unique(Y_true_train)
    )

    other args:
        cmap - the color map you want the confusion matrix chart to have.
        Other values can be 'flare'

        style - the background of the plot e.g. dark or light
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))
    axis = sb.heatmap(conf_matrix, cmap=cmap, annot=True, fmt='g')
    axis.set_title(img_title, )

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def view_metric_values(metrics_df, img_title: str="untitled", save_img: bool=True, colormap: str='mako', style: str='dark'):
    """
    given a each list of the training, validation, and testing set
    groups accuracy, precision, recall, and f1-score, plot a bar
    graph that separates these three groups metric values

    calculate accuracy, precision, recall, and f1-score for every 
    data split using the defined data_split_metric_values() function 
    above:

    train_acc, train_prec, train_rec, train_f1 = data_split_metric_values(Y_true_train, Y_pred_train)
    val_acc, val_prec, val_rec, val_f1 = data_split_metric_values(Y_true_val, Y_pred_val)
    test_acc, test_prec, test_rec, test_f1 = data_split_metric_values(Y_true_test, Y_pred_test)

    metrics_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'accuracy': [train_acc, val_acc, test_acc], 
        'precision': [train_prec, val_prec, test_prec], 
        'recall': [train_rec, val_rec, test_rec], 
        'f1-score': [train_f1, val_f1, test_f1]
    })
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # initialize empty array to be later converted to numpy
    colors = []
    
    # excludes the data split column
    n_metrics = metrics_df.shape[1] - 1
    rgb_colors = cm.get_cmap(colormap, n_metrics)
    for i in range(rgb_colors.N):
        rgb_color = rgb_colors(i)
        colors.append(str(mplt.colors.rgb2hex(rgb_color)))
    colors = np.array(colors)

    # sample n ids based on number of metrics of metrics df
    sampled_ids = np.random.choice(list(range(colors.shape[0])), size=n_metrics, replace=False)
    sampled_colors = colors[sampled_ids]

    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()

    # uses the given array of the colors you want to use
    sb.set_palette(sb.color_palette(sampled_colors))

    # create accuracy, precision, recall, f1-score of training group
    # create accuracy, precision, recall, f1-score of validation group
    # create accuracy, precision, recall, f1-score of testing group
    df_exp = metrics_df.melt(id_vars='data_split', var_name='metric', value_name='score')
    
    axis = sb.barplot(data=df_exp, x='data_split', y='score', hue='metric', ax=axis)
    axis.set_title(img_title, )
    axis.set_yscale('log')
    axis.legend()

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def view_classified_labels(df, img_title: str="untitled", save_img: bool=True, colors: list=['#db7f8e', '#b27392'], style: str='dark'):
    """
    given a each list of the training, validation, and testing set
    groups accuracy, precision, recall, and f1-score, plot a bar
    graph that separates these three groups metric values

    calculates all misclassified vs classified labels for training,
    validation, and testing sets by taking in a dataframe called
    classified_df created with the following code:

    num_right_cm_train = conf_matrix_train.trace()
    num_right_cm_val = conf_matrix_val.trace()
    num_right_cm_test = conf_matrix_test.trace()

    num_wrong_cm_train = train_labels.shape[0] - num_right_cm_train
    num_wrong_cm_val = val_labels.shape[0] - num_right_cm_val
    num_wrong_cm_test = test_labels.shape[0] - num_right_cm_test

    classified_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'classified': [num_right_cm_train, num_right_cm_val, num_right_cm_test], 
        'misclassified': [num_wrong_cm_train, num_wrong_cm_val, num_wrong_cm_test]}, 
        index=["training set", "validation set", "testing set"])
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()

    # uses the given array of the colors you want to use
    sb.set_palette(sb.color_palette(colors))

    # create classified and misclassified of training group
    # create classified and misclassified of validation group
    # create classified and misclassified of testing group
    df_exp = df.melt(id_vars='data_split', var_name='status', value_name='score')
    
    axis = sb.barplot(data=df_exp, x='data_split', y='score', hue='status', ax=axis)
    axis.set_title(img_title, )
    axis.legend()

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def view_label_freq(label_freq, img_title: str="untitled", save_img: bool=True, labels: list | pd.Series | np.ndarray=["DER", "NDG", "OFF", "HOM"], horizontal: bool=True, style: str='dark'):
    """
    suitable for all discrete input

    main args:
        label_freq - is actually a the returned value of the method
        of a pandas series, e.g.
            label_freq = df['label'].value_counts()
            label_freq

        labels - a list of all the labels we want to use in the 
        vertical bar graph
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # plots the unique labels against the count of these unique labels

    axis = sb.barplot(x=label_freq.values, y=labels, palette="flare") \
        if horizontal == True else sb.barplot(x=labels, y=label_freq.values, palette="flare")
    x_label = "frequency" if horizontal == True else "value"
    y_label = "value" if horizontal == True else "frequency"
    axis.set_xlabel(x_label, )
    axis.set_ylabel(y_label, )
    axis.set_title(img_title, )

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def disp_cat_feat(df, cat_cols: list, fig_dims: tuple=(3, 2), img_title: str="untitled", save_img: bool=True, style: str='dark'):
    """
    suitable for all discrete input

    displays frequency of categorical features of a dataframe
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # unpack dimensions of figure
    rows, cols = fig_dims
    
    # setup figure
    # fig, axes = plt.subplots(rows, cols, figsize=(15, 15), gridspec_kw={'width_ratios': [3, 3], 'height_ratios': [5, 5, 5]})
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), gridspec_kw={'width_ratios': [3, 3]})
    axes = axes.flat
    fig.tight_layout(pad=7)

    # helper function
    def hex_color_gen():
        rgb_gen = lambda: np.random.randint(0, 255)
        color = "#%02X%02X%02X" % (rgb_gen(), rgb_gen(), rgb_gen())
        return color

    # loop through all categorical features and see their frequencies
    for index, col in enumerate(cat_cols):
        # get value and respective counts of current categorical column
        val_counts = df[col].value_counts()

        # count the number of unique values
        n_unqiue = val_counts.shape[0]

        # get all unqiue categories of the feature/column
        keys = list(val_counts.keys())

        colors = [hex_color_gen() for _ in range(n_unqiue)]
        print(colors, n_unqiue)
        chosen_colors = np.random.choice(colors, n_unqiue, replace=False)
        # list all categorical columns no of occurences of each of their unique values
        ax = val_counts.plot(kind='barh', ax=axes[index], color=chosen_colors)

        # annotate bars using axis.containers[0] since it contains
        # all 
        print(ax.containers[0])
        ax.bar_label(ax.containers[0], )
        ax.set_ylabel('no. of occurences', )
        ax.set_xlabel(col, )
        ax.set_title(img_title, )
        ax.legend()

        # current column
        print(col)

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def plot_all_features(X, hue=None, colormap: str='mako', style: str='dark'):
    """
    suitable for: all discrete inputs, both discrete and continuous inputs,
    and all continuous inputs

    args:
        X - the dataset we want all features to be visualized whether
        discrete or continuous

        hue - a string that if provided will make the diagonals
        of the pairplot to be bell curves of the provided string feature
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    sb.set_palette(colormap)
    sb.pairplot(X, hue=hue, plot_kws={'marker': 'p', 'linewidth': 1})

# for recommendation
def describe_col(df: pd.DataFrame, column: str, style: str='dark'):
    """
    args:
        df - pandas data frame
        column - column of data frame to observe unique values and frequency of each unique value
    """

    print(f'count/no. of occurences of each unique {column} out of {df.shape[0]}: \n')

    unique_ids = df[column].unique()
    print(f'total unique values: {len(unique_ids)}')

def visualize_graph(kg, node_limit: str=500, edge: str='film.film.genre', node_color: str='skyblue', img_title: str="untitled", save_img: bool=True, style: str='dark'):
    """
    args:
        kg - is the knowledge graph represented as a dataframe with columns head, relation, tail
        which are the triples that make up the knowledge graph

        edge - the edges of the knowledge graph that will only be include in the
        visualization, for instance if there are unique relations under the relation
        column e.g. acted_in, directed, watched, etc. assigning 'directed' as the
        argument value will only show the graph with node to node connections having
        only the edge 'directed'

        e.g. 
        | head | relation | tail |
        | brad | acted_in | fury |
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))
    
    # see first 500 rows
    G = nx.from_pandas_edgelist(kg[:node_limit].loc[kg['relation'] == edge], source='head', target='tail', edge_attr=True, create_using=nx.MultiDiGraph())
    pos = nx.spring_layout(G, k=0.5)

    plt.figure(figsize=(12, 12))

    nx.draw(G, with_labels=True, node_color=node_color, edge_cmap=plt.cm.Blues, pos=pos)
    nx.draw_networkx_edge_labels(G, pos=pos)

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

class ModelResults:
    def __init__(self, history, epochs, style: str='dark'):
        """
        args:
            history - the history dictionary attribute extracted 
            from the history object returned by the self.fit() 
            method of the tensorflow Model object 

            epochs - the epoch list attribute extracted from the history
            object returned by the self.fit() method of the tensorflow
            Model object
        """
        self.history = history
        self.epochs = epochs
        self.style = style

    def _build_results(self, metrics_to_use: list):
        """
        builds the dictionary of results based on history object of 
        a tensorflow model

        returns the results dictionary with the format {'loss': 
        [24.1234, 12.1234, ..., 0.2134], 'val_loss': 
        [41.123, 21.4324, ..., 0.912]} and the number of epochs 
        extracted from the attribute epoch of the history object from
        tensorflow model.fit() method

        args:
            metrics_to_use - a list of strings of all the metrics to extract 
            and place in the dictionary
        """

        # extract the epoch attribute from the history object
        epochs = self.epochs
        results = {}
        for metric in metrics_to_use:
            if metric not in results:
                # extract the history attribute from the history object
                # which is a dictionary containing the metrics as keys, and
                # the the metric values over time at each epoch as the values
                results[metric] = self.history[metric]

        return results, epochs
    
    def export_results(self, dataset_id: str="untitled", metrics_to_use: list=['loss', 
                                            'val_loss', 
                                            'binary_crossentropy', 
                                            'val_binary_crossentropy', 
                                            'binary_accuracy', 
                                            'val_binary_accuracy', 
                                            'precision', 
                                            'val_precision', 
                                            'recall', 
                                            'val_recall', 
                                            'f1_m', 
                                            'val_f1_m', 
                                            'auc', 
                                            'val_auc',
                                            'categorical_crossentropy',
                                            'val_categorical_crossentropy'], save_img: bool=True):
        """
        args:
            metrics_to_use - a list of strings of all the metrics to extract 
            and place in the dictionary, must always be of even length
        """

        # extracts the dictionary of results and the number of epochs
        results, epochs = self._build_results(metrics_to_use)
        results_items = list(results.items())

        # we want to leave the user with the option to 
        for index in range(0, len(metrics_to_use) - 1, 2):
            # say 6 was the length of metrics to use
            # >>> list(range(0, 6 - 1, 2))
            # [0, 2, 4]
            metrics_indeces = (index, index + 1)
            curr_metric, curr_metric_perf = results_items[metrics_indeces[0]]
            curr_val_metric, curr_val_metric_perf = results_items[metrics_indeces[1]]
            print(curr_metric)
            print(curr_val_metric)
            curr_result = {
                curr_metric: curr_metric_perf,
                curr_val_metric: curr_val_metric_perf
            }
            print(curr_result)

            self.view_train_cross_results(
                results=curr_result,
                epochs=epochs, 
                curr_metrics_indeces=metrics_indeces,
                save_img=save_img,
                img_title="model performance using {} dataset for {} metric".format(dataset_id, curr_metric)
            )

    def view_train_cross_results(self, results: dict, epochs: list, curr_metrics_indeces: tuple, save_img: bool, img_title: str="untitled"):
        """
        plots the number of epochs against the cost given cost values 
        across these epochs.
        
        main args:
            results - is a dictionary created by the utility preprocessor
            function build_results()
        """
        styles = {
            'dark': 'dark_background',
            'solarized': 'Solarized_Light2',
            '538': 'fivethirtyeight',
            'ggplot': 'ggplot',
        }

        plt.style.use(styles.get(self.style, 'default'))

        figure = plt.figure(figsize=(15, 10))
        axis = figure.add_subplot()

        styles = [
            ('p:', '#f54949'), 
            ('h-', '#f59a45'), 
            ('o--', '#afb809'), 
            ('x:','#51ad00'), 
            ('+:', '#03a65d'), 
            ('8-', '#035aa6'), 
            ('.--', '#03078a'), 
            ('>:', '#6902e6'),
            ('p-', '#c005e6'),
            ('h--', '#fa69a3'),
            ('o:', '#240511'),
            ('x-', '#052224'),
            ('+--', '#402708'),
            ('8:', '#000000')]

        for index, (key, value) in enumerate(results.items()):
            # value contains the array of metric values over epochs
            # e.g. [213.1234, 123.43, 43.431, ..., 0.1234]

            if key == "loss" or key == "val_loss":
                # e.g. loss, val_loss has indeces 0 and 1
                # binary_cross_entropy, val_binary_cross_entropy 
                # has indeces 2 and 3
                axis.plot(
                    np.arange(len(epochs)), 
                    value, 
                    styles[curr_metrics_indeces[index]][0], 
                    color=styles[curr_metrics_indeces[index]][1], 
                    alpha=0.5, 
                    label=key, 
                    markersize=10, 
                    linewidth=3)
            else:
                # here if the metric value is not hte loss or 
                # validation loss each element is rounded by 2 
                # digits and converted to a percentage value 
                # which is why it is multiplied by 100 in order
                # to get much accurate depiction of metric value
                # that is not in decimal format
                metric_perc = [round(val * 100, 2) for val in value]
                axis.plot(
                    np.arange(len(epochs)), 
                    metric_perc, 
                    styles[curr_metrics_indeces[index]][0], 
                    color=styles[curr_metrics_indeces[index]][1], 
                    alpha=0.5, 
                    label=key, 
                    markersize=10, 
                    linewidth=3)

        # annotate end of lines
        for index, (key, value) in enumerate(results.items()):        
            if key == "loss" or key == "val_loss":
                last_loss_rounded = round(value[-1], 2)
                axis.annotate(last_loss_rounded, xy=(epochs[-1], value[-1]), color='black', alpha=1)
            else: 
                last_metric_perc = round(value[-1] * 100, 2)
                axis.annotate(last_metric_perc, xy=(epochs[-1], value[-1] * 100), color='black', alpha=1)

        axis.set_ylabel('metric value', )
        axis.set_xlabel('epochs', )
        axis.set_title(img_title, )
        axis.legend()

        if save_img == True:
            plt.savefig(f'./figures & images/{img_title}.png')
            plt.show()

        # delete figure
        del figure

def plot_evolution(X, K, centroids, xs_centroids, features: list, dimension='2d', style: str='dark'):
    """
    args:
        X - is the set of unlabeled datapoints of (m, n) dimensionality,
        where m represents the total number of all data points, and n
        is the number of features/columns/variables of each data point

        centroids - a 3D tensor of shape (epochs, K, n) that represents all 
        the previous centroids where K is the number of centroids, and n is
        the number of features of each centroid

        xs_centroids - is an array of m elements/indeces from 0 to K - 1
        representing the optimal and respective centroids of each data point. 
        To understand this say we had 3 cluster centroids, this array will
        have m elements and in each index it is assigned either a 0, 1, or 2
        depending if this index matching the set of unlabeled datapoints is
        indeed a part of either centroid 0, 1, or 2. E.g. [0, 1, 0, 2]
        means datapoint[0] is assigned as part of cluster centroid 0

        features - a list of all the names of each feature/column/variable
        of the dataset
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # define figure size
    fig = plt.figure(figsize=(11, 11))

    if dimension.lower() == '2d':
        axis = fig.add_subplot()
        axis.scatter(X[:, 0], X[:, 1], color='#90b2e8', marker='p',alpha=0.75,)

        for k in range(K):
            # gets all the centroids of cluster K at each epoch
            cs_of_k = centroids[:, k, :]
            m = cs_of_k.shape
            print(f'm: {m}')
            
            print(f'centroids of cluster {k}: {cs_of_k}\n')
            axis.plot(cs_of_k[:, 0], cs_of_k[:, 1], 'x--', alpha=0.25, color='black')
            axis.plot(cs_of_k[-1, 0], cs_of_k[-1, 1], 'p', color='#d60f7d')

    elif dimension.lower() == '3d':
        # 3d figure
        axis = fig.add_subplot(111, projection='3d')

        # sample color maps without replacement
        color_maps = ['viridis', 'magma', 'twilight', 'ocean', 'terrain', 'rainbow', 'gnuplot', 'RdPu', 'bone']
        chosen_colors = np.random.choice(color_maps, 3, replace=False)

        # iterate through all cluster centroid indeces
        for k in range(K):

            # extract all datapoints assigned to certain cluster
            cluster_k = X[xs_centroids == k]
            axis.scatter(cluster_k[:, 0], cluster_k[:, 1], cluster_k[:, 2], c=np.random.randn(cluster_k.shape[0]), marker='p',alpha=0.375, cmap=chosen_colors[k])

        for k in range(K):
            # get all the centroids of cluster K at each epoch
            cs_of_k = centroids[:, k, :]
            m = cs_of_k.shape
            print(f'm: {m}')
            
            print(f'centroids of cluster {k}: {cs_of_k}\n')
            axis.plot(cs_of_k[:, 0], cs_of_k[:, 1], cs_of_k[-1, 2], 'x--', color='black')
            axis.plot(cs_of_k[-1, 0], cs_of_k[-1, 1], cs_of_k[-1, 2], 'p--', color='#ff00bf')

        # n_clicks, amount_discount, amount_spent
        axis.set_xlabel(f'x: {features[0]}', )
        axis.set_ylabel(f'y: {features[1]}', )
        axis.set_zlabel(f'z: {features[2]}', )

def view_clusters_3d(X, features: list, style: str='dark'):
    """
    args:
        X - set of unlabeled data points of (m, n) dimensionality
        features - e.g. ['n_clicks', 'n_visits', 'amount_spent']
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[features[0]], X[features[1]], X[features[2]], c=np.random.randn(X.shape[0]), marker='p',alpha=0.75, cmap='magma')
    ax.set_xlabel(f'x: {features[0]}', )
    ax.set_ylabel(f'y: {features[1]}', )
    ax.set_zlabel(f'z: {features[2]}', )
    plt.show()

def view_images(data_gen, grid_dims: tuple=(2, 6), size: tuple=(25, 10), model=None, img_title="untitled", save_img: bool=True, style: str='dark'):
    """
    views images created by the ImageGenerator() class 
    from tensorflow.keras.preprocessing.image

    args: 
        data_gen - the data generator created from the ImageGenerator()
        method self.flow_from_directory()
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    class_names = list(data_gen.class_indices.keys())
    
    # The plotting configurations
    n_rows, n_cols = grid_dims
    n_images = n_rows * n_cols
    plt.figure(figsize=size)
    
    # gets a batch of the image data for visualization
    # This process can take a little time because of 
    # the large batch size
    images, labels = next(data_gen) 

    # sample n_images of indeces from array of indeces 
    # of length len(images) without replacement
    sampled_indeces = np.random.choice([num for num in range(len(images))], n_images, replace=False)

    # Iterate through the subplots.
    for i, id in enumerate(sampled_indeces, start=1):
        # use the randomly sampled id as index to access 
        # an image and its respective label
        image, label = images[id], class_names[np.argmax(labels[id], axis=0)]
        
        # Plot the sub plot
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(image)
        plt.axis('off')
        
        # If model is available make predictions
        if model is not None:
            pred = class_names[np.argmax(model.predict(image[np.newaxis,...]))]
            title = f"Class : {label}\nPred : {pred}"
        else:
            title = f"Class : {label}"
        
        plt.title(title, )
    
    if save_img == True:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()


def view_all_splits_results(history_dict: dict, save_img: bool=True, img_title: str="untitled", style: str='dark'):
    """
    
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }
    plt.style.use(styles.get(style, 'default'))

    # create the history dataframe using tensorflow history attribute
    history_df = pd.DataFrame(history_dict)
    print(history_df)

    palettes = np.array(['#f54949', '#f59a45', '#afb809', '#51ad00', '#03a65d', '#035aa6', '#03078a', '#6902e6', '#c005e6', '#fa69a3', '#240511', '#052224', '#402708', '#000000'])
    markers = np.array(['o', 'v', '^', '8', '*', 'p', 'h', ])#'x', '+', '>', 'd', 'H', '3', '4'])

    sampled_indeces = np.random.choice(list(range(len(markers))), size=history_df.shape[1], replace=False)

    print(palettes[sampled_indeces])
    print(markers[sampled_indeces])

    figure = plt.figure(figsize=(15, 10))
    axis = sb.lineplot(data=history_df, 
        palette=palettes[sampled_indeces].tolist(),
        markers=markers[sampled_indeces].tolist(), 
        linewidth=3.0,
        markersize=9,
        alpha=0.75)
    
    axis.set_ylabel('metric value', )
    axis.set_xlabel('epochs', )
    axis.set_title(img_title, )
    axis.legend()

    if save_img == True:
        print(save_img)
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def show_image(img, style: str='dark'):
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    plt.imshow(img)