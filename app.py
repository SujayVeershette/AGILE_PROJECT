import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
import sqlite3
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from scipy.sparse import issparse

# Initialize Streamlit app
st.title("📊 Multi-Model Classifier & Clustering App")

# Database functions
def get_df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def init_db():
    conn = sqlite3.connect("ml_results.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            df_hash TEXT,
            model TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            conf_matrix TEXT,
            per_class_precision TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_metrics_to_db(df_original, model_name, metrics_dict, conf_matrix, per_class_precision):
    df_hash = get_df_hash(df_original)
    conn = sqlite3.connect("ml_results.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO models_metrics (
            df_hash, model, accuracy, precision, recall, f1, conf_matrix, per_class_precision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        df_hash,
        model_name,
        metrics_dict['accuracy'],
        metrics_dict['precision'],
        metrics_dict['recall'],
        metrics_dict['f1_score'],
        json.dumps(conf_matrix.tolist()),
        json.dumps(per_class_precision.tolist())
    ))
    conn.commit()
    conn.close()

def load_metrics_if_exist(df_original, model_name):
    df_hash = get_df_hash(df_original)
    conn = sqlite3.connect("ml_results.db")
    cursor = conn.cursor()
    cursor.execute('''
        SELECT accuracy, precision, recall, f1, conf_matrix, per_class_precision
        FROM models_metrics
        WHERE df_hash=? AND model=?
        LIMIT 1
    ''', (df_hash, model_name))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'accuracy': row[0],
            'precision': row[1],
            'recall': row[2],
            'f1_score': row[3],
            'conf_matrix': np.array(json.loads(row[4])),
            'per_class_precision': np.array(json.loads(row[5]))
        }
    return None

# Initialize database
init_db()

# File upload and data processing
uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 *Preview of Data*", df.head())

    # Target selection
    target_column = st.selectbox("🎯 Select target column", df.columns)
    
    # Feature selection
    st.subheader("🔘 Select Features for Modeling")
    all_features = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect(
        "Choose features to include in X", 
        all_features, 
        default=all_features
    )
    
    if not selected_features:
        st.error("Please select at least one feature!")
        st.stop()

    X = df[selected_features]
    y = df[target_column]

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Feature processing functions
    def is_text_column(series):
        if series.dtype == object:
            sample = series.dropna().head(10)
            return any(isinstance(x, str) and any(c.isalpha() for c in str(x)) for x in sample)
        return False

    def is_categorical_text(series):
        unique_count = series.dropna().nunique()
        return 1 < unique_count < 20

    def is_categorical_numeric(series):
        return pd.api.types.is_numeric_dtype(series) and 1 < series.nunique() < 20

    # Categorize columns
    text_cols = [col for col in X.columns if is_text_column(X[col])]
    cat_num_cols = [col for col in X.columns if is_categorical_numeric(X[col]) and col not in text_cols]
    pure_num_cols = [col for col in X.select_dtypes(include=np.number).columns 
                    if col not in cat_num_cols and col not in text_cols]

    final_features = []
    feature_info = []

    # Process text columns
    for col in text_cols:
        col_data = X[col].fillna('')
        if is_categorical_text(col_data):
            le = LabelEncoder()
            X[col] = le.fit_transform(col_data)
            final_features.append(X[[col]])
            feature_info.append(f"Text (categorical - label encoded): {col}")
        else:
            tfidf = TfidfVectorizer(max_features=100)
            tfidf_result = tfidf.fit_transform(col_data)
            tfidf_df = pd.DataFrame(tfidf_result.toarray(), 
                                  columns=[f"{col}_{word}" for word in tfidf.get_feature_names_out()])
            final_features.append(tfidf_df)
            feature_info.append(f"Text (TF-IDF vectorized): {col} ({tfidf_df.shape[1]} features)")

    # Process numeric categorical columns
    if cat_num_cols:
        num_cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', MinMaxScaler())
        ])
        num_cat_data = num_cat_transformer.fit_transform(X[cat_num_cols])
        num_cat_df = pd.DataFrame(num_cat_data, columns=cat_num_cols)
        final_features.append(num_cat_df)
        feature_info.append(f"Numeric categorical (min-max scaled): {', '.join(cat_num_cols)}")

    # Process pure numeric columns
    if pure_num_cols:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        numeric_data = numeric_transformer.fit_transform(X[pure_num_cols])
        numeric_df = pd.DataFrame(numeric_data, columns=pure_num_cols)
        final_features.append(numeric_df)
        feature_info.append(f"Continuous numeric (standard scaled): {', '.join(pure_num_cols)}")

    # Combine all features
    try:
        X_all_transformed_df = pd.concat(final_features, axis=1)
        numeric_cols = pure_num_cols + cat_num_cols
        
        # Display feature processing summary
        st.subheader("🔍 Feature Processing Summary")
        for info in feature_info:
            st.write(f"✅ {info}")
            
    except Exception as e:
        st.error(f"❌ Error combining features: {str(e)}")
        st.stop()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all_transformed_df, y_encoded, test_size=0.2, random_state=42)

    # Define models
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Naïve Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='rbf', C=1.0, random_state=42)
    }

    accuracy_scores = {}

    # Model training and evaluation
    for name, model in models.items():
        st.subheader(f"🤖 Model: {name}")
        
        cached_metrics = load_metrics_if_exist(df, name)
        
        if cached_metrics:
            st.write(f"📊 Using cached results (from previous run)")
            st.write(f"*Accuracy:* {cached_metrics['accuracy']*100:.2f}%")
            st.write(f"*Precision:* {cached_metrics['precision']*100:.2f}%")
            st.write(f"*Recall:* {cached_metrics['recall']*100:.2f}%")
            st.write(f"*F1 Score:* {cached_metrics['f1_score']*100:.2f}%")
            accuracy_scores[name] = cached_metrics['accuracy'] * 100
            
            # Confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cached_metrics['conf_matrix'], annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_,
                        ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix - {name}")
            st.pyplot(fig)
            
        else:
            # Prepare data based on model requirements
            X_train_model = X_train.values if not issparse(X_train) else X_train.toarray()
            X_test_model = X_test.values if not issparse(X_test) else X_test.toarray()
            
            if name == "Naïve Bayes":
                X_train_model = X_train_model.astype('float')
                X_test_model = X_test_model.astype('float')
            
            # Train model
            model.fit(X_train_model, y_train)
            y_pred = model.predict(X_test_model)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            per_class_precision = precision_score(y_test, y_pred, average=None)
            
            metrics_dict = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
            # Save to database
            save_metrics_to_db(df, name, metrics_dict, conf_matrix, per_class_precision)
            accuracy_scores[name] = accuracy * 100
            
            # Display metrics
            st.write(f"*Accuracy:* {accuracy*100:.2f}%")
            st.write(f"*Precision:* {metrics_dict['precision']*100:.2f}%")
            st.write(f"*Recall:* {metrics_dict['recall']*100:.2f}%")
            st.write(f"*F1 Score:* {metrics_dict['f1_score']*100:.2f}%")
            
            # Per-class metrics
            st.write("\n*Per-Class Metrics:*")
            metrics_df = pd.DataFrame({
                'Class': label_encoder.classes_,
                'Precision': precision_score(y_test, y_pred, average=None),
                'Recall': [report[str(i)]['recall'] for i in range(len(label_encoder.classes_))],
                'F1-Score': [report[str(i)]['f1-score'] for i in range(len(label_encoder.classes_))]
            })
            st.dataframe(metrics_df.style.format({
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}'
            }))
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_,
                        ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix - {name}")
            st.pyplot(fig)

    # Model comparison
    st.subheader("📈 Model Accuracy Comparison")
    comparison_df = pd.DataFrame.from_dict(accuracy_scores, orient='index', columns=['Accuracy'])
    st.bar_chart(comparison_df)
    
    # Clustering
    st.subheader("🧠 KMeans Clustering")
    num_clusters = st.slider("Select number of clusters (K)", 2, 10, 3)
    
    if len(numeric_cols) >= 2:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_data = X_all_transformed_df[numeric_cols].values
        kmeans.fit(cluster_data)
        df["Cluster"] = kmeans.labels_
        
        st.write("🔍 *Clustered Data Preview*")
        st.write(df.head())
        
        # Cluster visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            cluster_data[:, 0], 
            cluster_data[:, 1], 
            c=df["Cluster"], 
            cmap="viridis",
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.title(f"KMeans Clustering (K={num_clusters})")
        st.pyplot(fig)
        
        # Cluster statistics
        st.write("\n*Cluster Statistics:*")
        cluster_stats = df.groupby("Cluster")[numeric_cols].mean()
        st.dataframe(cluster_stats.style.format("{:.2f}"))
    else:
        st.warning("⚠ Need at least 2 numeric features for clustering visualization")
