

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

URL = 'https://raw.githubusercontent.com/jacklong233/ST1/main/Rice_MSC_Dataset_Trimmed.csv'
@st.cache_resource
def load_data():
    df = pd.read_csv(URL)

    for col in df:
        if df[col].dtype == 'object':
            df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))

    df_normalized = (df - df.min()) / (df.max() - df.min())

    le = LabelEncoder()
    labels = le.fit_transform(df_normalized['CLASS'])
    df_normalized.drop('CLASS', axis=1, inplace=True)
    return df_normalized, labels

def main():
    st.title("Rice Data Classifier")

    data, target = load_data()

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose the Classifier",
        ("DecisionTree", "NaiveBayes", "SVM", "GradientBoosting", "RandomForest"))

    if st.sidebar.button("Train Model"):
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=7)

        if model_choice == "DecisionTree":
            model = DecisionTreeClassifier()
        elif model_choice == "NaiveBayes":
            model = GaussianNB()
        elif model_choice == "SVM":
            model = SVC(probability=True)
        elif model_choice == "GradientBoosting":
            model = GradientBoostingClassifier()
        else:
            model = RandomForestClassifier()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        st.write(f"{model_choice} Accuracy: {accuracy_score(y_test, y_pred)}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap='Blues')
        st.pyplot(fig)

        y_prob = model.predict_proba(x_test)[:, 0]
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=0)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_title('ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        st.pyplot(fig)


if __name__ == "__main__":
    main()
