#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time
from sklearn.pipeline import make_pipeline
scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

#%%
###############################################################################
#               Toy dataset : 2 gaussians
###############################################################################

n1 = 200 # Nombre de points dans la première classe
n2 = 200 # Nombre de points dans la deuxieme classe
mu1 = [1., 1.] # moyenne de la première gaussienne
mu2 = [-1./2, -1./2] # moyenne de la deuxième gaussienne
sigma1 = [0.9, 0.9] # écart-type de la première gaussienne
sigma2 = [0.9, 0.9] # écart-type de la deuxième gaussienne
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

plt.style.use('default')
plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 5))
plot_2d(X1, y1)


X_train = X1[::2]
Y_train = y1[::2].astype(int)
X_test = X1[1::2]
Y_test = y1[1::2].astype(int)

# fit the model with linear kernel
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)


# predict labels for the test data base
y_pred = clf.predict(X_test)


# check your score
score = clf.score(X_test, Y_test)
print('Score : %s' % score)

# display the frontiere
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

# Same procedure but with a grid search
parameters = {'kernel': ['linear'], 'C': list(np.linspace(0.001, 3, 21))}
clf2 = SVC()
clf_grid = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_grid.fit(X_train, Y_train)

# check your score
print(clf_grid.best_params_)
print('Score : %s' % clf_grid.score(X_test, Y_test))

def f_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_grid.predict(xx.reshape(1, -1))

# display the frontiere
plt.figure()
frontiere(f_grid, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
###############################################################################
#               Iris Dataset
###############################################################################

iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]


# split train test (say 25% for the test)
# You can shuffle and then separate or you can just use train_test_split
# whithout shuffling (in that case fix the random state (say to 42) for reproductibility)


# la consigne demande de split en 2 alors que dans le script on utilise 25%...
# j'ai choisi 25% pour rester cohérent avec le reste du script

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)



###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################


# plot des données avec les vraies classes
plt.figure()
plot_2d(X, y)
plt.draw()

#%%
# Q1 Linear kernel

# fit the model and select the best hyperparameter C
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}
clf_linear = GridSearchCV(estimator=SVC(),param_grid=parameters,cv=5,n_jobs=-1)
clf_linear.fit(X_train, y_train)

# compute the score
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

#%%
# Q2 polynomial kernel
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[3] # j'enlève le 1 car c'est un kernel linéaire et on l'a deja fait

# fit the model and select the best set of hyperparameters
parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_poly = GridSearchCV(
    estimator=SVC(),
    param_grid=parameters,
    cv=5,
    n_jobs=-1
)
clf_poly.fit(X_train, y_train)

print(clf_poly.best_params_)
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))


#%%
# display your results using frontiere (svm_source.py)


def f_linear(xx):
    # xx vecteur 2D
    return clf_linear.predict(xx.reshape(1, -1))[0]

def f_poly(xx):
    return clf_poly.predict(xx.reshape(1, -1))[0]

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()


#%%
# plot des figures 1 à 1
frontiere(f_poly, X, y)

#%%
###############################################################################
#               SVM GUI
###############################################################################

# please open a terminal and run python svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel


#%%
###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

#%%
# Q4
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []


for C in Cs:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))

plt.style.use('default')
plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()


#%%
# predict labels for the X_test images with the best classifier
best_C = Cs[ind]
clf = SVC(kernel='linear', C=best_C)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))

#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()


#%%
# Q5

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X, y)

print("Score avec variable de nuisance")
n_features = X.shape[1]
# On rajoute des variables de nuisances
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, )
#with gaussian coefficients of std sigma
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
run_svm_cv(X_noisy, y)

#%%
# Q6
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Normalisation des données
scaler = StandardScaler()
X_noisy_normalized = scaler.fit_transform(X_noisy)

# Application de la PCA
pca = PCA()
pca.fit(X_noisy_normalized)

# Extraction des valeurs propres (variance expliquée par composante)
explained_variance_ratio = pca.explained_variance_ratio_

# Création de l'histogramme
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center', label='Variance expliquée')
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Proportion de variance expliquée')
plt.title('Éboulis des valeurs propres (Histogramme)')
plt.legend()
plt.grid()
plt.xlim(0.5, 25.5)
plt.tight_layout()
plt.show()

#%%

print("Score apres reduction de dimension")

# IL FAUT NORMALISER AVANT DE FAIRE LA PCA sinon ca compile pas

scaler = StandardScaler()
X_noisy_normalized = scaler.fit_transform(X_noisy)

n_components = 5
pca = PCA(n_components=n_components)
X_noisy_pca = pca.fit_transform(X_noisy_normalized)

_parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 3))}  # Réduisez les valeurs de C

run_svm_cv(X_noisy_pca, y)
# %%
#######################################################################################################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.25, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)  # Fit sur l'ensemble d'entraînement
X_test_normalized = scaler.transform(X_test)       # Transforme l'ensemble de test

# Application de la PCA
n_components = 20
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_normalized)  # Fit sur l'ensemble d'entraînement
X_test_pca = pca.transform(X_test_normalized)        # Transforme l'ensemble de test

# Entraînement du modèle SVM
clf = SVC(kernel='linear', C=1)  # Vous pouvez ajuster le paramètre C
clf.fit(X_train_pca, y_train)

# Calcul et affichage des scores
train_score = clf.score(X_train_pca, y_train)
test_score = clf.score(X_test_pca, y_test)

print("Score sur l'ensemble d'entraînement :", train_score)
print("Score sur l'ensemble de test :", test_score)
# %%
