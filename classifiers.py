import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC
import joblib



## Uso: python script.py <nome_do_arquivo> <num_features>
##
## 
##
if len(sys.argv) != 3:
    print("Uso: python script.py <nome_do_arquivo>")
else:
    pathFeatures = sys.argv[1]
    pathFeatures = pathFeatures[2: ] # tira o ./
    print(pathFeatures)
    numFeatures  = int(sys.argv[2])
    data = np.loadtxt(pathFeatures, dtype='str') # , delimiter=','
    X = data[: , 0 : -numFeatures]
    y = data[: , -numFeatures]
    fileResult = ("resultados_" + pathFeatures)
    


models = './models/'


extractor = 'VIT'


classes = np.unique(y)
print(classes)


modelo = 'svm'


def fit_svm_stratified(X_train, y_train, iteration):
    
    modelo = "svm"
    svc = SVC(probability=True)
    
    param_grid = {
        'C': [0.001, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(estimator = svc, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 3) #, scoring = 'f1_weighted') 


    best_model = grid_search.fit(X_train, y_train)

    joblib.dump(best_model,models+'best_'+ extractor +"_"+modelo+"_"+str(iteration)+'.pkl')

    return best_model

# gera a predição e a matriz de confusão
def predict_stratified(model, X_test, y_test, iteration):
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-score:", f1)
    
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n=== MATRIZ DE CONFUSÃO (Fold", iteration, ") ===\n")
    df_cm = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    print(df_cm)
    print("\n")

    f = open(fileResult,"a")
    f.write("%f \n" % (f1))
    f.close()
    
    return f1



def fit_stratified_count(X,y):
    f1_scores = []
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, y)
    j = 1
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]
        print(len(X_test))

        j+=1
    return j


def fit_stratified(X,y):
    f1_scores = []
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, y)
    j = 1
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]
        
        model = fit_svm_stratified(X_train, y_train, j)

        f1 = predict_stratified(model, X_test, y_test, j)
        f1_scores.append(f1)
        
        j+=1
    return f1_scores


columns = ["nome","média original", "std original","tudo"]

dados = pd.DataFrame(columns=columns)


sc = StandardScaler()


f1_weighted_scores = fit_stratified_count(X,y)
media = round(np.mean(f1_weighted_scores)*100,2)
std = round(np.std(f1_weighted_scores)*100,2)

new_data = pd.Series({"nome" : extractor+"_" +modelo,"média original" : media,
                      "std original" : std, "tudo" : f"{str(media)} +- {str(std)}"})

new_index = len(dados)
dados.loc[new_index] = new_data



f1_weighted_scores = []
f1_weighted_scores = fit_stratified(X,y)


media = round(np.mean(f1_weighted_scores)*100,2)
std = round(np.std(f1_weighted_scores)*100,2)

f = open(fileResult,"a")

f.write("\n%f \n" % (media))
f.write("\n%f \n" % (std))

f.close()
