# %% [markdown]
# Streamlit app for KNN Lab

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib.colors import ListedColormap

# Set page config
st.set_page_config(page_title="KNN Lab", layout="wide")

# Load data (cached)
@st.cache_data
def load_data():
    data = np.loadtxt('dataset.dat', skiprows=1)
    X = data[:, 0:2]
    y = data[:, 2].astype(int)
    return X, y

X, y = load_data()

# Sidebar controls
st.sidebar.header("Parameters")
test_size = st.sidebar.slider("Test size ratio", 0.1, 0.5, 0.3)
max_k = st.sidebar.slider("Maximum k for analysis", 5, 50, 20)

# Train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42)

# Main content
st.title("KNN Classification Lab")
st.markdown("---")

# Section 1: Data visualization
st.header("1. Data Visualization")
fig1 = plt.figure(figsize=(6, 4))
colors = np.array([x for x in "rgbcmyk"])
plt.scatter(X[:, 0], X[:, 1], color=colors[y].tolist(), s=10)
plt.title("Dataset Visualization")
st.pyplot(fig1)

# Section 2: Training size impact
st.header("2. Impact of Training Set Size")
subset_sizes = np.linspace(0.01, 1.0, 20)
num_samples = [int(size * len(X_train)) for size in subset_sizes]

@st.cache_data
def compute_learning_curve():
    accuracies = []
    for size in num_samples:
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train[:size], y_train[:size])
        accuracies.append(knn.score(X_test, y_test))
    return accuracies

learning_acc = compute_learning_curve()

fig2 = plt.figure(figsize=(8, 4))
plt.plot(num_samples, learning_acc, marker='o')
plt.xlabel("Training samples")
plt.ylabel("Test accuracy")
plt.title("Learning Curve")
st.pyplot(fig2)

# Section 3: KNN Analysis
st.header("3. KNN Analysis")
col1, col2 = st.columns(2)

with col1:
    k = st.slider("Select k value", 1, max_k, 8)
    
    # Decision boundary plot
    def plot_decision(k):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        h = .02
        x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
        y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
        plt.title(f"Decision Boundaries (k={k})")
        return plt.gcf()

    fig3 = plot_decision(k)
    st.pyplot(fig3)

with col2:
    # Accuracy metrics
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    
    st.metric("Training Accuracy", f"{train_acc:.2%}")
    st.metric("Test Accuracy", f"{test_acc:.2%}")
    
    # Confusion matrix
    y_pred = knn.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig4 = plt.figure()
    metrics.ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    st.pyplot(fig4)

# Section 4: Cross-validation
st.header("4. Cross-validation Analysis")

@st.cache_data
def compute_cv_scores():
    k_values = range(1, max_k+1)
    cv_scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = model_selection.cross_val_score(knn, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())
    return cv_scores

cv_acc = compute_cv_scores()
optimal_k = np.argmax(cv_acc) + 1

fig5 = plt.figure(figsize=(8, 4))
plt.plot(range(1, max_k+1), cv_acc, marker='o')
plt.xlabel("k value")
plt.ylabel("CV Accuracy")
plt.title(f"Optimal k: {optimal_k} (Cross-validation)")
st.pyplot(fig5)

st.markdown("---")
st.info("Adjust parameters in the sidebar to explore different configurations!")