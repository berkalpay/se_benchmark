from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from models import DrugClust, FS_MLKNN, CS

baseline_models = [
    ("SMC", MultiOutputClassifier(DummyClassifier(strategy="prior"))),
    ]
generic_models = [
    ("kNN", KNeighborsClassifier(n_neighbors=55, metric="jaccard")),
    ("RF", RandomForestClassifier(n_estimators=80, random_state=42)),
    ("SVM", MultiOutputClassifier(SVC(gamma="auto", probability=True, random_state=42), n_jobs=-1)),
    ]
custom_models = [
    ("DrugClust", DrugClust()),
    ("CS", CS("data/CS/ADR_similarity/")),
    ("MLKNN", FS_MLKNN(feature_selection=False)),
    ]
