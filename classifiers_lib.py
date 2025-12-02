# classifiers_lib.py
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# CLASSIFICADORES BASE

def get_base_classifiers():

    clfs = []

    # KNN com diferentes valores de k
    for k in [1, 3, 5, 7, 9]:
        clfs.append((f"knn_{k}", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))))

    # Árvores de Decisão com diferentes profundidades
    depths = [None, 5, 10, 20, 30]
    for d in depths:
        name = "dt_null" if d is None else f"dt_{d}"
        clfs.append((name, DecisionTreeClassifier(max_depth=d, random_state=42)))

    # SVM com diferentes kernels e valores de C
    for kernel in ['rbf', 'linear']:
        for C in [0.1, 1.0, 10.0]:
            
            clfs.append((f"svm_{kernel}_C{C}", make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, probability=True, random_state=42))))

    # MLP com diferentes neurônios
    mlp_variants = [
        (100,), (100,50), (200,), (200,100), (300,)
    ]
    for hid in mlp_variants:
        clfs.append((f"mlp_{'_'.join(map(str,hid))}", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=hid, max_iter=500, random_state=42))))

    # Random Forest com diferentes números de árvores
    for n in [50, 100, 200, 300]:
        clfs.append((f"rf_{n}", RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=42)))

    return clfs


# voting ensembles
def get_voting_ensemble(classifier_list, voting='hard'):
    chosen = classifier_list[:20]
    return VotingClassifier(estimators=chosen, voting=voting, n_jobs=-1)

# stacking ensemble
def get_stacking_ensemble(classifier_list, final_estimator=None):
    chosen = classifier_list[:20]
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000)
    return StackingClassifier(estimators=chosen, final_estimator=final_estimator, n_jobs=-1)
