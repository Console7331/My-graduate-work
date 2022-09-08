import random
from tkinter import *
import pandas as pd
import numpy as np
#import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from umap import UMAP
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def version1():
    window = Tk()
    window.title("Дипломная работа Суровцева Р.В. ИМММО-01-21")
    window.mainloop()

def version2():
    data = pd.read_csv('Coca_Cola_Company.csv')
    data.isna().sum()
    close = data['Close']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=1/close,mode='lines', line=dict(color='blue', width=1.5)))
    #fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    #fig.add_vline(x=1571, line_width=1, line_dash="dash", line_color="red")#e^2
    fig.update_layout(  height=720, 
                        width=700,
                        autosize=False,
                        #plot_bgcolor="white",
                        yaxis_title='1/Close',
                        xaxis_title='t, торговые дни',
                        showlegend=False)
    
def version3():
    def read_transactions(filename: str):
        with open(filename, mode="r", newline='') as f:
            lines = list()
            for row in f:
                line = list()
                for s in row.split(';'):
                    s = s.strip()
                    if s:
                        line.append(s)
                lines.append(line)
        return lines[1:]
    transactions = read_transactions("Coca_Cola_Company.csv")
    time_apriori = 22.21
    time_e_apriori = 1.32
    all_colors = list(plt.cm.colors.cnames.keys())
    c = random.choices(all_colors, k=2)
    plt.figure(figsize=(5, 5))
    plt.bar(["time_apriori", "time_e_apriori"], [time_apriori, time_e_apriori], color=c, width=.5)
    plt.text(0, time_apriori, float(time_apriori), horizontalalignment='center', verticalalignment='bottom')
    plt.text(1, time_e_apriori, float(time_e_apriori), horizontalalignment='center', verticalalignment='bottom')
    plt.ylim(0, time_apriori+1)
    plt.title("1000 repeats of transactions")
    plt.show()

def version4():
    def plot_embeddings(embedded_tsne, embedded_umap, targets):
        labels = list(range(np.max(targets)+1))
        palette = np.array(sns.color_palette(n_colors=len(labels)))
        patchs = []
        for i, color in enumerate(palette):
            patchs.append(mpatches.Patch(color=color, label=i))
        plt.figure(figsize=(16, 16))
        plt.subplot(2, 1, 1)
        plt.scatter(embedded_tsne[:,0], embedded_tsne[:,1], c=palette[targets])
        plt.legend(handles=patchs, loc='upper right')
        plt.title("Embedded with t-SNE")
        plt.subplot(2, 1, 2)
        plt.scatter(embedded_umap[:,0], embedded_umap[:,1], c=palette[targets])
        plt.legend(handles=patchs, loc='upper right')
        plt.title("Embedded with UMAP")
        plt.show()
    def read_data(filename: str, delimeter=';'):
        with open(filename, mode="r", newline='') as f:
            features = list()
            targets = list()
            for row in f:
                line = list()
                for s in row.split(delimeter):
                    s = s.strip()
                    if s:
                        line.append(s)
                features.append(list(map(float ,line[1:-1])))
                targets.append(int(line[-1]))
        return features, targets
    features, targets = read_data('/content/hayes-roth.data', delimeter=',')
    set(targets)
    embedded_tsne = TSNE().fit_transform(features)
    embedded_umap = UMAP().fit_transform(features)
    plot_embeddings(embedded_tsne, embedded_umap, targets)
    min_max_scaled_tsne = TSNE().fit_transform(MinMaxScaler().fit_transform(features))
    min_max_scaled_umap = UMAP().fit_transform(MinMaxScaler().fit_transform(features))
    plot_embeddings(min_max_scaled_tsne, min_max_scaled_umap, targets)
    standard_scaled_tsne = TSNE().fit_transform(StandardScaler().fit_transform(features))
    standard_scaled_umap = UMAP().fit_transform(StandardScaler().fit_transform(features))
    plot_embeddings(standard_scaled_tsne, standard_scaled_umap, targets)
    robust_scaled_tsne = TSNE().fit_transform(RobustScaler().fit_transform(features))
    robust_scaled_umap = UMAP().fit_transform(RobustScaler().fit_transform(features))
    plot_embeddings(robust_scaled_tsne, robust_scaled_umap, targets)

def version5():
    def plot_results(features, targets, model):
        def make_meshgrid(x, y, h=.02):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            return xx, yy
        def plot_contours(model, xx, yy, **params):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            out = plt.contourf(xx, yy, Z, **params)
            return out
        xx, yy = make_meshgrid(features[:, 0], features[:, 1])
        plt.figure(figsize=(20, 10))
        colormap = 'coolwarm'
        labels = np.unique(targets).tolist()
        palette = np.array(sns.color_palette(colormap, n_colors=len(labels)))
        cmap = sns.color_palette(colormap, as_cmap=True)
        patchs = []
        for i, color in enumerate(palette):
            patchs.append(mpatches.Patch(color=color, label=i))
        plot_contours(model, xx, yy, cmap=cmap, alpha=0.8)
        plt.scatter(features[:, 0], features[:, 1], c=targets, cmap=cmap, s=40, edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis('off')
        # plt.legend(handles=patchs, loc='upper right')
        plt.show()
    def read_data(filename: str, delimeter=';'):
        with open(filename, mode="r", newline='') as f:
            features = list()
            targets = list()
            for row in f:
                line = list()
                for s in row.split(delimeter):
                    s = s.strip()
                    if s:
                        line.append(s)
                
                t = int(line[-1])
                if t < 3:
                    t -= 1
                    features.append(list(map(float ,line[1:-1])))
                    targets.append(t)
        return features, targets
    features, targets = read_data('hayes-roth.data', delimeter=',')
    features = np.array(features)
    features = UMAP().fit_transform(features)
    features = StandardScaler().fit_transform(features)
    targets = np.array(targets)
    np.unique(targets)
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, stratify=targets, shuffle=True)
    x_train.shape, x_test.shape
    svc_parameters = {
        'kernel':['poly'],
        'C': np.linspace(1, 10, 11),
        'degree': np.linspace(2, 15, 14)
    }
    clf_svc = GridSearchCV(SVC(), svc_parameters)
    clf_svc.fit(x_train, y_train)
    clf_svc.best_params_
    print(classification_report(y_train, clf_svc.predict(x_train), digits=3, zero_division=0))
    print(classification_report(y_test, clf_svc.predict(x_test), digits=3, zero_division=0))
    best_svc = SVC(**clf_svc.best_params_)
    best_svc.fit(x_train, y_train)
    len(best_svc.support_vectors_)
    plot_results(features, targets, best_svc)
