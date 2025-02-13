# %% [markdown]
# # TP1: Algorithme des plus proches voisins

# %% [markdown]
# ## 1) Analyse des données

# %%
import numpy as np
import plotly
import matplotlib 
import sklearn
from matplotlib import pyplot

# %%
data = np.loadtxt('dataset.dat', skiprows=1)

# %%
data

# %%
#séparer les observations et les labels 

X = data[:,0:2] 
y = data[:,2] 
y = y.astype(int)

# %%
from matplotlib import pyplot 
colors = np.array([x for x in "rgbcmyk"]) 
pyplot.scatter(X[:, 0], X[:, 1], color=colors[y].tolist(), s=10) 
pyplot.show()


# %%
#partition des données
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, 
test_size=0.3)
#choisir et entraîner un modèle (KNN)
from sklearn.neighbors import KNeighborsClassifier
one_NN = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
one_NN.fit(X_train, y_train)
#score sur la base d'apprentissage
print('accuraccy on training set:', one_NN.score(X_train, y_train))
#matrice de confusion
from sklearn import metrics
y_pred_test = one_NN.predict(X_test)
metrics.confusion_matrix(y_test, y_pred_test)

# %%
# Créer une grille
x_min, x_max = X[:, 0].min()*1.1, X[:, 0].max()*1.1
y_min, y_max = X[:, 1].min()*1.1, X[:, 1].max()*1.1
x_h = (x_max - x_min)/50
y_h = (y_max - y_min)/50
xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h),
np.arange(y_min, y_max, y_h))
Y = one_NN.predict(np.c_[xx.ravel(), yy.ravel()])
Y = Y.reshape(xx.shape)
#afficher les frontières/données d'apprentissage
pyplot.contourf(xx, yy, Y, cmap=pyplot.cm.Paired, alpha=0.8)
pyplot.scatter(X_train[:, 0], X_train[:, 1], cmap=pyplot.cm.Paired, color=colors[y_train].tolist())
pyplot.xlim(xx.min(), xx.max())
pyplot.ylim(yy.min(), yy.max())
pyplot.show()

# %%


# %% [markdown]
# ## 3)Analyse du fonctionnement de l’algorithme

# %% [markdown]
# # 🔬 Protocole d'Analyse : Impact de la Taille d'Apprentissage
# 
# ## 📌 Objectif de l'Expérience
# 
# ### 🎯 But Principal
# - Évaluer l'impact de la taille des données d'apprentissage sur 1-NN
# - Identifier le point optimal d'apprentissage
# - Comprendre la relation taille/performance
# 
# ### 📊 Paramètres d'Étude
# - **Algorithme** : 1-NN (k=1)
# - **Données** : X_train1 (base d'apprentissage)
# - **Validation** : X_test (base de test fixe)
# - **Plage d'étude** : 1% à 100% de X_train1
# 
# ## 🔍 Protocole Expérimental
# 
# ### 📈 Étapes d'Analyse
# 1. **Préparation des Données**
#    - Générer des sous-ensembles de X_train1
#    - Échantillonnage : 1% → 100%
#    - Maintenir la distribution des classes
# 
# 2. **Expérimentation**
#    - Entraîner 1-NN sur chaque sous-ensemble
#    - Évaluer sur X_test complet
#    - Mesurer la précision
# 
# 3. **Visualisation**
#    - Tracer la courbe d'apprentissage
#    - Axe X : Taille de l'échantillon
#    - Axe Y : Taux de reconnaissance
# 
# ## 💡 Questions de Recherche
# 
# ### 🔍 Points d'Analyse
# 1. **Evolution de la Performance**
#    - Comment évolue la précision ?
#    - Y a-t-il des paliers ?
#    - Quand apparaît la convergence ?
# 
# 2. **Seuil de Stabilité**
#    - Nombre minimal d'exemples nécessaire
#    - Point de stabilisation
#    - Rapport coût/bénéfice
# 
# ### 📊 Métriques à Observer
# - Taux de reconnaissance
# - Variabilité des résultats
# - Points de rupture dans la courbe
# 
# ## 🎯 Résultats Attendus
# 
# ### 📈 Observations Anticipées
# - Progression initiale rapide
# - Plateau de performance
# - Identification du seuil optimal
# 
# ### 💭 Implications
# - Optimisation de la taille d'apprentissage
# - Compromis ressources/performance
# - Recommandations pratiques
# 

# %%
# Recharger la base de données après le nouveau téléversement
file_path = "dataset.dat"
data = np.loadtxt(file_path)

# Séparer les observations (features) et les labels
X = data[:, 0:2]  # Les deux premières colonnes sont les features
y = data[:, 2].astype(int)  # La dernière colonne est le label, converti en entier

# Vérification des dimensions
X.shape, y.shape


# %%
# Re-diviser la base en apprentissage (70%) et test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Définir les tailles des sous-ensembles de X_train1 (de 1% à 100%)
subset_sizes_train = np.linspace(0.01, 1.0, 20)  # 20 valeurs entre 1% et 100%
num_samples_train = [int(size * len(X_train)) for size in subset_sizes_train]  # Convertir en nombre d'exemples

# Stocker les taux de reconnaissance
accuracy_scores_train = []

for size in num_samples_train:
    # Créer un sous-ensemble de la base d’apprentissage
    X_train1 = X_train[:size]
    y_train1 = y_train[:size]
    
    # Initialiser et entraîner le modèle 1-NN
    one_NN = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    one_NN.fit(X_train1, y_train1)
    
    # Évaluer la performance sur la base de test complète X_test
    accuracy = one_NN.score(X_test, y_test)
    accuracy_scores_train.append(accuracy)

# Tracer le graphe du taux de reconnaissance en fonction du nombre d'exemples d'apprentissage
plt.figure(figsize=(10, 5))
plt.plot(num_samples_train, accuracy_scores_train, marker='o', linestyle='-')
plt.xlabel("Nombre d'exemples d'apprentissage")
plt.ylabel("Taux de reconnaissance sur X_test")
plt.title("Impact du nombre d'exemples d'apprentissage sur la précision")
plt.grid()
plt.show()

# Afficher les premiers et derniers résultats pour analyse
accuracy_scores_train[:5], accuracy_scores_train[-5:]


# %% [markdown]
# # 📊 Analyse de l'Impact de la Taille des Données d'Apprentissage
# 
# ## 📈 Evolution de la Précision
# 
# ### 🔍 1% des Données (2 exemples)
# - 📉 Précision très faible : ~44%
# - ⚠️ Apprentissage insuffisant
# - ❌ Généralisation médiocre
# 
# ### 📊 10-20% des Données
# - 📈 Augmentation rapide de la précision
# - 🎯 Performance : ~85-91%
# - ✨ Amélioration significative
# 
# ### 📈 70%+ des Données (~150 exemples)
# - 🎯 Stabilisation : 82-85%
# - 📊 Plateau de performance
# - ⚖️ Point d'équilibre optimal
# 
# ## 🔍 Analyse Détaillée
# 
# ### 📉 Phase Initiale (Peu d'Exemples)
# - ❌ Faible capacité de généralisation
# - ⚠️ Modèle peu fiable
# - 📊 Performance insuffisante
# 
# ### 📈 Phase d'Apprentissage
# - ✅ Amélioration progressive
# - 📊 Réduction des erreurs
# - 🎯 Meilleure généralisation
# 
# ### 🔄 Phase de Plateau
# - 📊 Seuil atteint à ~70% des données
# - ⚖️ Plus d'exemples n'améliore pas significativement la performance
# - 🎯 Point optimal d'efficacité
# 
# ## 💡 Conclusions Clés
# 
# ### 📌 Points Critiques
# 1. **Données Minimales**
#    - ⚠️ 1% est insuffisant
#    - ❌ Généralisation impossible
#    
# 2. **Zone Optimale**
#    - ✅ 70% des données suffisent
#    - 🎯 Balance coût/performance optimale
# 
# 3. **Loi des Rendements Décroissants**
#    - 📊 Au-delà de 70%, gain marginal
#    - ⚖️ Coût additionnel non justifié
# 
# ## 🎯 Recommandations
# - ✅ Utiliser au moins 20% des données pour un apprentissage viable
# - 🎯 Viser 70% pour une performance optimale
# - 💡 Ne pas surcharger inutilement au-delà du plateau
# 

# %%
# Définir les tailles des sous-ensembles de X_test1 (de 1% à 100%)
subset_sizes_test = np.linspace(0.01, 1.0, 20)  # 20 valeurs entre 1% et 100%
num_samples_test = [max(1, int(size * len(X_test))) for size in subset_sizes_test]  # S'assurer que la taille min est 1

# Stocker les taux de reconnaissance
accuracy_scores_test = []

# Utilisation de toute la base d'apprentissage X_train
for size in num_samples_test:
    # Créer un sous-ensemble de la base de test
    X_test1 = X_test[:size]
    y_test1 = y_test[:size]
    
    # Initialiser et entraîner le modèle 1-NN sur toute la base d’apprentissage
    one_NN = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    one_NN.fit(X_train, y_train)
    
    # Évaluer la performance sur la base de test réduite X_test1
    accuracy = one_NN.score(X_test1, y_test1)
    accuracy_scores_test.append(accuracy)

# Tracer le graphe du taux de reconnaissance en fonction du nombre d'exemples de test
plt.figure(figsize=(10, 5))
plt.plot(num_samples_test, accuracy_scores_test, marker='o', linestyle='-')
plt.xlabel("Nombre d'exemples de test")
plt.ylabel("Taux de reconnaissance sur X_test1")
plt.title("Impact du nombre d'exemples de test sur la précision")
plt.grid()
plt.show()

# Afficher les premiers et derniers résultats pour analyse
accuracy_scores_test[:5], accuracy_scores_test[-5:]


# %% [markdown]
# # 📊 Analyse de l'Impact de k sur la Précision
# 
# ## 🎯 Valeur Optimale de k (k*)
# 
# ### ✨ Résultats Clés
# - 🏆 Meilleure précision : k* = 8
# - ⚖️ Représente l'équilibre optimal entre biais et variance
# - 🎯 8 voisins = point optimal pour la prise de décision
# 
# ## 📈 Analyse des Tendances
# 
# ### 🔍 k = 1 : Zone de Surapprentissage
# - 📉 Biais très faible
# - 📈 Variance très élevée
# - ⚠️ Modèle trop spécialisé aux données d'entraînement
# 
# ### 📊 Evolution avec k Croissant
# 1. **Phase d'Amélioration**
#    - 📈 La précision augmente
#    - 🎯 Tend vers k* (8)
#    - ✅ Meilleur équilibre progressif
# 
# 2. **Phase de Détérioration**
#    - 📉 La précision diminue après k*
#    - ⚠️ Modèle devient trop général
#    - 🔍 Augmentation du biais
# 
# ### ⚠️ Impact d'un k Trop Grand
# - 🔸 Lissage excessif des frontières de décision
# - 📉 Perte de capacité à capturer les structures complexes
# - ❌ Sous-apprentissage du modèle
# 
# ## 💡 Conclusion
# L'analyse montre clairement que k=8 représente le point optimal où :
# - ✅ Le modèle généralise bien
# - ✅ Les frontières de décision sont suffisamment flexibles
# - ✅ Le compromis biais-variance est optimal
# 

# %% [markdown]
# ## 4)Algorithme des k-ppv

# %%
# Déterminer une valeur raisonnable pour kmax
kmax = int(np.sqrt(len(X_train)))  # Une règle souvent utilisée est sqrt(N) où N est le nombre d'exemples d'apprentissage
k_values = range(1, kmax + 1)  # Tester k de 1 à kmax

# Stocker les performances
accuracy_scores_k = []

for k in k_values:
    # Initialiser et entraîner le modèle k-NN
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    knn.fit(X_train, y_train)
    
    # Évaluer la performance sur la base de test
    accuracy = knn.score(X_test, y_test)
    accuracy_scores_k.append(accuracy)

# Tracer le graphe du taux de reconnaissance en fonction de k
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracy_scores_k, marker='o', linestyle='-')
plt.xlabel("Valeur de k")
plt.ylabel("Taux de reconnaissance sur X_test")
plt.title("Impact du paramètre k sur la précision en test")
plt.grid()
plt.show()

# Trouver la valeur optimale de k*
optimal_k = k_values[np.argmax(accuracy_scores_k)]
optimal_k


# %% [markdown]
# # 📊 Analyse de l'Impact de k sur la Précision
# 
# ## 🎯 Valeur Optimale de k (k*)
# 
# ### ✨ Résultats Clés
# - 🏆 Meilleure précision : k* = 8
# - ⚖️ Représente l'équilibre optimal entre biais et variance
# - 🎯 8 voisins = point optimal pour la prise de décision
# 
# ## 📈 Analyse des Tendances
# 
# ### 🔍 k = 1 : Zone de Surapprentissage
# - 📉 Biais très faible
# - 📈 Variance très élevée
# - ⚠️ Modèle trop spécialisé aux données d'entraînement
# 
# ### 📊 Evolution avec k Croissant
# 1. **Phase d'Amélioration**
#    - 📈 La précision augmente
#    - 🎯 Tend vers k* (8)
#    - ✅ Meilleur équilibre progressif
# 
# 2. **Phase de Détérioration**
#    - 📉 La précision diminue après k*
#    - ⚠️ Modèle devient trop général
#    - 🔍 Augmentation du biais
# 
# ### ⚠️ Impact d'un k Trop Grand
# - 🔸 Lissage excessif des frontières de décision
# - 📉 Perte de capacité à capturer les structures complexes
# - ❌ Sous-apprentissage du modèle
# 
# ## 💡 Conclusion
# L'analyse montre clairement que k=8 représente le point optimal où :
# - ✅ Le modèle généralise bien
# - ✅ Les frontières de décision sont suffisamment flexibles
# - ✅ Le compromis biais-variance est optimal
# 

# %%
# Définition d'une fonction pour tracer les frontières de décision
from matplotlib.colors import ListedColormap

def plot_decision_boundaries(k, X_train, y_train, title):
    # Initialiser le modèle k-NN
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    knn.fit(X_train, y_train)

    # Création de la grille pour la visualisation
    h = .02  # Taille de la maille du grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Prédictions sur la grille
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Affichage des frontières et des points
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', alpha=0.7)
    plt.title(f"Frontières de décision pour k = {k}")
    plt.show()

# Afficher les frontières pour k = 1, k* et kmax
plot_decision_boundaries(1, X_train, y_train, "k = 1 (Fort variance, faible biais)")
plot_decision_boundaries(optimal_k, X_train, y_train, f"k = {optimal_k} (Équilibre optimal)")
plot_decision_boundaries(kmax, X_train, y_train, f"k = {kmax} (Fort biais, faible variance)")


# %%
# Stocker les performances sur la base d'apprentissage
accuracy_scores_train_set = []

for k in k_values:
    # Initialiser et entraîner le modèle k-NN
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    knn.fit(X_train, y_train)
    
    # Évaluer la performance sur la base d'apprentissage
    accuracy = knn.score(X_train, y_train)
    accuracy_scores_train_set.append(accuracy)

# Tracer le graphe du taux de reconnaissance en fonction de k (apprentissage)
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracy_scores_train_set, marker='o', linestyle='-', label="Apprentissage", color='blue')
plt.plot(k_values, accuracy_scores_k, marker='s', linestyle='--', label="Test", color='red')
plt.xlabel("Valeur de k")
plt.ylabel("Taux de reconnaissance")
plt.title("Impact du paramètre k sur la précision en apprentissage et en test")
plt.legend()
plt.grid()
plt.show()

# Afficher les premiers et derniers résultats pour analyse
accuracy_scores_train_set[:5], accuracy_scores_train_set[-5:]


# %% [markdown]
# # 🎯 K-Nearest Neighbors: Decision Boundary Analysis
# 
# ## 🔍 Analysis by K Value
# 
# ### 🎯 K=1: High Variance Region
# > *Maximum Flexibility, Minimum Stability*
# 
# **📊 Key Features**
# - 🔸 Highly complex, irregular boundaries
# - 🔸 Single-neighbor dependency
# - 🔸 Perfect training data fit
# 
# **⚠️ Challenges**
# - 🚫 Extreme noise sensitivity
# - 🚫 Poor generalization capability
# 
# **📝 Summary:** Maximum flexibility but unstable predictions
# 
# ---
# 
# ### 🎯 K=8: Optimal Balance
# > *Sweet Spot between Flexibility and Stability*
# 
# **📊 Key Features**
# - ✅ Balanced boundary smoothness
# - ✅ Reduced noise sensitivity
# - ✅ Stable predictions
# 
# **💪 Strengths**
# - 🎯 Excellent generalization
# - 🎯 Robust predictions
# 
# **📝 Summary:** Optimal performance with balanced bias-variance trade-off
# 
# ---
# 
# ### 🎯 K=14: High Bias Region
# > *Maximum Stability, Minimum Flexibility*
# 
# **📊 Key Features**
# - 📉 Over-smoothed boundaries
# - 📉 Reduced model flexibility
# - 📉 Detail loss at boundaries
# 
# **⚠️ Limitations**
# - 🚫 Underfitting risk
# - 🚫 Loss of important patterns
# 
# **📝 Summary:** Too stable, missing important patterns
# 
# ---
# 
# ## 🎓 Impact of K Selection
# 
# | K Value | Bias | Variance | Characteristic |
# |---------|------|----------|----------------|
# | Small (1-3) | ⬇️ Low | ⬆️ High | Overfitting |
# | Optimal (8) | ➡️ Balanced | ➡️ Balanced | Best Performance |
# | Large (14+) | ⬆️ High | ⬇️ Low | Underfitting |
# 
# ## 🎯 Conclusion
# The optimal value K=8 achieves the perfect balance between:
# - 🎯 Model flexibility
# - 🎯 Prediction stability
# - 🎯 Generalization capability
# 
# This provides the best trade-off between bias and variance for robust predictions.
# 

# %% [markdown]
# ### **Analyse de l'impact de \( k \) sur la précision en apprentissage et en test (Étape 4d)**  
# 
# #### **1️⃣ Tendance générale du graphique :**  
# - La **courbe bleue (apprentissage)** représente le taux de reconnaissance sur la base d'entraînement.  
# - La **courbe rouge (test)** représente le taux de reconnaissance sur la base de test.  
# 
# 🔹 **Pour \( k = 1 \) :**  
#    - **Apprentissage** : La précision est **100%** → **Le modèle mémorise parfaitement les données**.  
#    - **Test** : La précision est faible (~82%) → **Le modèle surapprend et ne généralise pas bien**.  
# 
# 🔹 **Quand \( k \) augmente :**  
#    - **En apprentissage** : La précision **diminue progressivement** car le modèle devient plus lisse et ne s’adapte plus aux moindres variations.  
#    - **En test** : La précision **augmente d'abord** (moins d'erreurs dues au bruit) mais **se stabilise autour de 85%** après \( k = 8 \).  
# 
# 🔹 **Pour \( k \) très grand (\( k = 14 \)) :**  
#    - **Apprentissage** : La précision reste assez élevée mais **moins flexible**.  
#    - **Test** : La précision **se stabilise**, car le modèle est devenu **trop généraliste**.  
# 
# ---
# 
# ### **2️⃣ Explication de la relation entre biais et variance**  
# | **Valeur de \( k \)** | **Biais** | **Variance** | **Explication** |
# |---|---|---|---|
# | **\( k = 1 \)** | Très faible | Très forte | Le modèle **mémorise tout**, y compris le bruit (surapprentissage). |
# | **\( k = k^* = 8 \)** | Modéré | Modéré | **Bon équilibre** entre généralisation et précision. |
# | **\( k = k_{max} = 14 \)** | Fort | Faible | Le modèle devient **trop général** et ne capture plus assez de détails. |
# 
# ---
# 
# ### **3️⃣ Comparaison entre les courbes Apprentissage vs Test**
# - **Quand \( k \) est petit**, le modèle **surdéveloppe** sur les données d'entraînement mais **ne généralise pas bien** sur de nouvelles données.  
# - **Quand \( k \) est optimal (\( k^* = 8 \))**, les performances en **test sont maximales** et les performances en apprentissage restent **élevées mais réalistes**.  
# - **Quand \( k \) est trop grand**, la précision en **apprentissage diminue**, mais celle en test **se stabilise**.
# 
# ---
# 
# ### **4️⃣ Conclusion sur le choix de \( k \)**
# ✅ **\( k^* = 8 \) est la meilleure valeur** :  
# - **Équilibre entre biais et variance**  
# - **Meilleure généralisation**  
# - **Performance stable en test et apprentissage**  
# 
# 📌 **Prochaine étape (4e) : Évaluer le protocole utilisé pour régler \( k \)**  
# 💡 **Question d’analyse :**  
# - **Notre méthode est-elle satisfaisante ?**
# - **Peut-on l'améliorer (validation croisée, autre métrique) ?**  
# 
# Veux-tu qu'on analyse cette dernière partie ? 🚀

# %%
from sklearn.model_selection import cross_val_score

# Définir les valeurs de k à tester
k_values = range(1, kmax + 1)

# Stocker les scores de validation croisée
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')  # Validation croisée 5-fold
    cv_scores.append(scores.mean())

# Trouver la meilleure valeur de k selon la validation croisée
optimal_k_cv = k_values[np.argmax(cv_scores)]

# Tracer le graphe du taux de reconnaissance en fonction de k (validation croisée vs test simple)
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracy_scores_k, marker='o', linestyle='-', label="Test unique", color='red')
plt.plot(k_values, cv_scores, marker='s', linestyle='--', label="Validation croisée", color='blue')
plt.xlabel("Valeur de k")
plt.ylabel("Taux de reconnaissance")
plt.title("Comparaison entre validation croisée et test unique")
plt.legend()
plt.grid()
plt.show()

# Comparer les résultats
optimal_k, optimal_k_cv


# %% [markdown]
# ### **Analyse de la validation croisée vs test unique pour le choix de \( k \) (Étape 4e)**  
# 
# #### **1️⃣ Interprétation du graphique**  
# - **La courbe rouge (test unique)** représente le taux de reconnaissance en test selon la méthode initiale.
# - **La courbe bleue (validation croisée 5-fold)** représente la moyenne des précisions obtenues après validation croisée.
# 
# 🔹 **Différences observées :**  
#    - La validation croisée **donne une courbe plus lisse et plus stable**.  
#    - La méthode de test unique **présente plus de fluctuations** (elle dépend fortement de la séparation spécifique train/test).  
#    - **Pour \( k \leq 4 \)**, les deux courbes se ressemblent, mais **au-delà, la validation croisée donne de meilleurs résultats globaux**.  
# 
# 🔹 **Meilleur choix de \( k \) avec la validation croisée :**  
#    - **\( k^*_{\text{test unique}} = 8 \)** (méthode initiale).  
#    - **\( k^*_{\text{validation croisée}} = 10 \)** (choix plus stable).  
#    - Avec la validation croisée, \( k^* \) est légèrement plus élevé, ce qui peut **réduire la variance** sans trop augmenter le biais.
# 
# ---
# 
# #### **2️⃣ Pourquoi la validation croisée est meilleure ?**  
# ✅ **Avantages :**
#    - **Moins dépendante d'une seule séparation train/test**.
#    - **Plus fiable pour généraliser sur des données nouvelles**.
#    - **Réduit l'effet de surapprentissage** en lissant les variations.
# 
# ❌ **Limites :**
#    - **Plus coûteuse en temps de calcul**, car on entraîne et teste plusieurs fois.
#    - **Peut ne pas être nécessaire si le dataset est très grand** (car la séparation simple est déjà assez représentative).
# 
# ---
# 
# ### **3️⃣ Conclusion sur le protocole de choix de \( k \)**
# | **Méthode** | **Simplicité** | **Fiabilité** | **Stabilité des résultats** | **Coût computationnel** |
# |-------------|--------------|--------------|-------------------|------------------|
# | **Test unique** | ✅ Très simple | ⚠️ Dépend du split train/test | ⚠️ Fluctuations visibles | ✅ Rapide |
# | **Validation croisée** | ⚠️ Plus complexe | ✅ Plus fiable | ✅ Résultats plus stables | ⚠️ Plus lent |
# 
# 📌 **Recommandation :**  
# - **Si le dataset est petit/modéré** → **Validation croisée préférable**.  
# - **Si le dataset est très grand** → La méthode simple peut suffire.  
# - Dans notre cas, la validation croisée donne une **meilleure estimation** et nous recommande **\( k^* = 10 \)** au lieu de **\( k^* = 8 \)**.
# 
# ---
# 
# ### **4️⃣ Conclusion finale du choix de \( k \) pour \( k \)-NN**
# ✅ **Le protocole basé sur la validation croisée est plus fiable** car :
#    - Il **généralise mieux** les performances du modèle.
#    - Il évite **les fluctuations dues à un unique split train/test**.
#    - Il aide à **choisir un \( k^* \) plus robuste**.
# 
# 💡 **Prochaine étape possible : Expérimenter des optimisations pour accélérer \( k \)-NN, comme KD-Tree ou Ball-Tree.**  
# 

# %%



