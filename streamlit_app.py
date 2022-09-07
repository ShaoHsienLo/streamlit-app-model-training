import random
import pandas as pd
import numpy as np
import streamlit as st
import io
import pickle

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scikitplot.metrics import plot_confusion_matrix, plot_roc_curve, plot_cumulative_gain, plot_ks_statistic, \
    plot_precision_recall, plot_lift_curve
from scikitplot.estimators import plot_learning_curve, plot_feature_importances
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from imblearn.combine import SMOTEENN

st.set_page_config(layout="wide", page_title="Title")

# 加入css
with open('style.css') as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


class ShowPerformance:

    def __init__(self, model, X_train_res, X_test_norm, y_train_res, y_test, target_names=None,
                 target_original_value=None, save_plot=False, model_name=""):
        self.model = model
        self.X_train_res = X_train_res
        self.X_test_norm = X_test_norm
        self.y_train_res = y_train_res
        self.y_test = y_test
        self.target_names = target_names
        self.target_original_value = target_original_value
        self.model_name = model_name
        self.save_plot = save_plot

        self.train_set_score = self.model.score(self.X_train_res, self.y_train_res)
        self.test_set_score = self.model.score(self.X_test_norm, self.y_test)
        self.train_y_pred = self.model.predict(X_train_res)
        self.test_y_pred = self.model.predict(X_test_norm)
        self.train_y_probs = self.model.predict_proba(X_train_res)
        self.test_y_probs = self.model.predict_proba(X_test_norm)

    def run(self):
        with st.expander("Other results", expanded=True):

            self.target_names_mapping()
            self.score()
            self.confution_matrix()
            self.roc_curve()
            self.ks_statistic()
            self.precision_recall()
            self.cumulative_gain()
            self.lift_curve()

            self.learning_curve()
            if self.model_name in ["rf", "xgboost"]:
                self.feature_importances()

    def target_names_mapping(self):
        st.write("Target names mapping")
        labels = {}
        for val, lab in zip(self.target_original_value, self.target_names):
            labels[str(val)] = lab
        st.write(labels)

    def score(self):
        st.write("Score")
        score = roc_auc_score(self.y_test, self.test_y_probs[:, 1])
        score_df = pd.DataFrame(
            [[self.train_set_score, self.test_set_score, score]],
            columns=["train set", "test set", "roc auc"]
        )
        st.write(score_df)

    def confution_matrix(self):
        st.write("Confusion Metric")
        plot_confusion_matrix(y_true=self.y_test, y_pred=self.test_y_pred, normalize=True)
        if self.save_plot:
            plt.savefig("./results/{}_confution_matrix.svg".format(self.model_name))
        st.pyplot()

    def roc_curve(self):
        st.write("ROC Curve")
        plot_roc_curve(y_true=self.y_test, y_probas=self.test_y_probs)
        if self.save_plot:
            plt.savefig("./results/{}_roc_curve.svg".format(self.model_name))
        st.pyplot()

    def ks_statistic(self):
        st.write("KS Statistic")
        plot_ks_statistic(y_true=self.y_test, y_probas=self.test_y_probs)
        if self.save_plot:
            plt.savefig("./results/{}_ks_statistic.svg".format(self.model_name))
        st.pyplot()

    def precision_recall(self):
        st.write("Precision-Recall Curve")
        plot_precision_recall(y_true=self.y_test, y_probas=self.test_y_probs)
        if self.save_plot:
            plt.savefig("./results/{}_precision_recall.svg".format(self.model_name))
        st.pyplot()

    def cumulative_gain(self):
        st.write("Cumulative Gain")
        plot_cumulative_gain(y_true=self.y_test, y_probas=self.test_y_probs)
        if self.save_plot:
            plt.savefig("./results/{}_cumulative_gain.svg".format(self.model_name))
        st.pyplot()

    def lift_curve(self):
        st.write("Lift Curve")
        plot_lift_curve(y_true=self.y_test, y_probas=self.X_test_norm)
        if self.save_plot:
            plt.savefig("./results/{}_lift_curve.svg".format(self.model_name))
        st.pyplot()

    def learning_curve(self):
        st.write("Learning Curve")
        plot_learning_curve(clf=self.model, X=self.X_train_res, y=self.y_train_res)
        if self.save_plot:
            plt.savefig("./results/{}_learning_curve.svg".format(self.model_name))
        st.pyplot()

    def feature_importances(self):
        # importrances = {"feature": self.X_test_norm.columns, "importance": self.model.feature_importances_}
        # importrances_df = pd.DataFrame(data=importrances).sort_values(by=["importance"], ascending=False)
        # st.write(importrances_df)

        # st.write(
        #     alt.Chart(importrances_df, width=1000).mark_bar().encode(
        #         x=alt.X("feature", sort=None),
        #         y="importance",
        #     )
        # )

        st.write("Feature Importances")
        plot_feature_importances(clf=self.model, feature_names=self.X_test_norm.columns, x_tick_rotation=90)
        if self.save_plot:
            plt.savefig("./results/{}_feature_importances.svg".format(self.model_name))
        st.pyplot()


class ModelTraining:

    def __init__(self, X_train_res, X_test, y_train_res, y_test, **kwargs):
        self.X_train_res = X_train_res
        self.X_test = X_test
        self.y_train_res = y_train_res
        self.y_test = y_test
        self.params = kwargs

    def save_pickle_model(self, model):
        model_data = io.BytesIO()
        pickle.dump(model, model_data)
        return model_data

    def auto(self):
        st.write("auto")
        pass

    def random_forest(self):
        rf = RandomForestClassifier(**self.params)
        rf.fit(self.X_train_res, self.y_train_res)
        model_data = self.save_pickle_model(rf)
        return rf, model_data

    def xgboost(self):
        xgboost = XGBClassifier(**self.params)
        xgboost.fit(self.X_train_res, self.y_train_res)
        model_data = self.save_pickle_model(xgboost)
        return xgboost, model_data

    def logistic_regression(self):
        lr = LogisticRegression()
        lr.fit(self.X_train_res, self.y_train_res)
        model_data = self.save_pickle_model(lr)
        return lr, model_data


class DatePreproccessing:

    def __init__(self, dataframe, **kwargs):
        self.dataframe = dataframe
        self.params = kwargs

    def run(self):
        df_handle_na = self.handle_na_values()
        df_encoded, target_names, target_original_value = self.handle_categorical_label(df_handle_na)
        df_selected = self.select_k_best(df_encoded)
        X_train, X_test, y_train, y_test = self.train_test_split(df_selected)
        X_train_norm, X_test_norm = self.normalization(X_train, X_test)
        X_train_res, y_train_res = self.smoteenn(X_train_norm, y_train)
        return X_train_res, X_test_norm, y_train_res, y_test, target_names, target_original_value

    def handle_na_values(self):
        if self.params["fill_na_method"] != "None":
            df_handle_na = self.dataframe.fillna(method=self.params["fill_na_method"])
            return df_handle_na
        return self.dataframe

    def handle_categorical_label(self, df_handle_na):
        labelencoder = LabelEncoder()
        df_handle_na[self.params["label"]] = labelencoder.fit_transform(df_handle_na[self.params["label"]])
        df_encoded = df_handle_na.copy()
        target_names = labelencoder.classes_
        target_original_value = df_handle_na[self.params["label"]].unique()
        return df_encoded, target_names, target_original_value

    def select_k_best(self, df_encoded):
        X = df_encoded.drop(columns=[self.params["label"]])
        y = df_encoded[self.params["label"]]
        fs = SelectKBest(score_func=f_classif, k=self.params["select_k_best"])
        fs.fit(X, y)
        mask = fs.get_support()
        features_selected = []
        for bool, col in zip(mask, X.columns):
            if bool:
                features_selected.append(col)
        df_selected = pd.concat([self.dataframe[features_selected], y], axis=1)
        return df_selected

    def train_test_split(self, df_selected):
        X = df_selected.drop(columns=[self.params["label"]])
        y = df_selected[self.params["label"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.params["test_size"], stratify=y,
                                                            random_state=self.params["random_state"])
        return X_train, X_test, y_train, y_test

    def normalization(self, X_train, X_test):
        if self.params["normalization"] == "Standard":
            scale = StandardScaler()
        elif self.params["normalization"] == "Min Max":
            scale = MinMaxScaler()
        X_train_norm = pd.DataFrame(scale.fit_transform(X_train), columns=X_train.columns)
        X_test_norm = pd.DataFrame(scale.transform(X_test), columns=X_test.columns)
        return X_train_norm, X_test_norm

    def smoteenn(self, X_train_norm, y_train):
        if self.params["smoteenn"] == "Not do":
            return X_train_norm, y_train
        elif self.params["smoteenn"] == "Do":
            sm = SMOTEENN()
            X_train_res, y_train_res = sm.fit_resample(X_train_norm, y_train)
            X_train_res = pd.DataFrame(X_train_res, columns=X_train_norm.columns)
            return X_train_res, y_train_res


uploaded_file = st.file_uploader("Upload file")

if uploaded_file is not None:

    dataframe = pd.read_csv(uploaded_file)
    dataframe = dataframe.drop(columns=["original label"])
    dataframe = dataframe.dropna()
    cols = dataframe.columns

    with st.expander("Preprocessing parameters", expanded=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            select_k_best = st.select_slider("select_k_best", options=list(np.arange(2, len(dataframe.columns), 1)),
                                             value=10 if len(dataframe.columns) > 10 else len(dataframe.columns))

        with col2:
            normalization = st.select_slider("normalization", options=("Standard", "Min Max"),
                                             value="Standard")

        with col3:
            smoteenn = st.select_slider("smoteenn", options=("Not do", "Do"), value="Not do")

        col1, col2, col3 = st.columns(3)

        with col1:
            random_state = st.slider("random state", 1, 100, random.randint(1, 100), 1)

        with col2:
            test_size = st.slider("test size", 0.1, 0.5, 0.2, 0.1)

        with col3:
            label = st.select_slider("label", options=cols, value=cols[-1])

        col1, col2, col3 = st.columns(3)

        with col1:
            fill_na_method = st.select_slider("fill na method", options=("None", "bfill", "ffill"), value="None")

    preprocessing_params = {
        "select_k_best": select_k_best,
        "normalization": normalization,
        "smoteenn": smoteenn,
        "random_state": random_state,
        "test_size": test_size,
        "label": label,
        "fill_na_method": fill_na_method
    }

    algorithms = ["automl", "random forest", "xgboost", "logistic regression"]
    option = st.selectbox("Algorithms", algorithms, index=0)

    if option == "automl":

        if st.button("Start training!"):
            st.write('No support automl')
            # model_training = ModelTraining(dataframe, label, test_size, **params)
            # model_training.auto()

    elif option == "random forest":

        data_preproccessing = DatePreproccessing(dataframe, **preprocessing_params)
        X_train_res, X_test_norm, y_train_res, y_test, target_names, target_original_value = data_preproccessing.run()

        with st.expander("Model parameters", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                n_estimators = st.slider("n estimators", 10, 1000, 100, 10)

            with col2:
                criterion = st.select_slider("criterion", options=("gini", "entropy"), value="gini")

            with col3:
                max_depth = st.slider("max depth", 1, 10, 3, 1)

        model_params = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "random_state": random_state,
        }

        save_plot = st.checkbox("Save plot?")
        start_training = st.button("Start training!")

        if start_training:
            model_training = ModelTraining(X_train_res, X_test_norm, y_train_res, y_test, **model_params)
            model, model_data = model_training.random_forest()
            show_performances = ShowPerformance(model, X_train_res, X_test_norm, y_train_res, y_test,
                                                target_names, target_original_value, save_plot, model_name="rf")
            show_performances.run()
            st.download_button("Download model", data=model_data, file_name="rf-model.pkl")
    elif option == "xgboost":

        data_preproccessing = DatePreproccessing(dataframe, **preprocessing_params)
        X_train_res, X_test_norm, y_train_res, y_test, target_names, target_original_value = data_preproccessing.run()

        with st.expander("Model parameters", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                n_estimators = st.slider("n estimators", 10, 1000, 100, 10)

            with col2:
                max_depth = st.slider("max depth", 1, 10, 3, 1)

            with col3:
                learning_rate = st.slider("learning rate", 0.01, 1.0, 0.1, 0.01)

        model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state
        }

        save_plot = st.checkbox("Save plot?")
        start_training = st.button("Start training!")

        if start_training:
            model_training = ModelTraining(X_train_res, X_test_norm, y_train_res, y_test, **model_params)
            model, model_data = model_training.xgboost()
            show_performances = ShowPerformance(model, X_train_res, X_test_norm, y_train_res, y_test,
                                                target_names, target_original_value, save_plot, model_name="xgboost")
            show_performances.run()
            st.download_button("Download model", data=model_data, file_name="xgboost-model.pkl")

    elif option == "logistic regression":

        data_preproccessing = DatePreproccessing(dataframe, **preprocessing_params)
        X_train_res, X_test_norm, y_train_res, y_test, target_names, target_original_value = data_preproccessing.run()

        with st.expander("Model parameters", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                penalty = st.select_slider("penalty", options=("l1", "l2", "elasticnet", "none"), value="l2")

            with col2:
                C = st.slider("C", 0.01, 1.0, 1.0, 0.01)

            with col3:
                max_iter = st.slider("max_iter", 10, 1000, 100, 10)

            col1, col2, col3 = st.columns(3)

            with col1:
                class_weight = st.select_slider("class_weight", options=("None", "balanced"), value="None")
                if class_weight == "None":
                    class_weight = None

        model_params = {
            "penalty": penalty,
            "C": C,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "random_state": random_state
        }

        save_plot = st.checkbox("Save plot?")
        start_training = st.button("Start training!")

        if start_training:
            model_training = ModelTraining(X_train_res, X_test_norm, y_train_res, y_test, **model_params)
            model, model_data = model_training.logistic_regression()
            show_performances = ShowPerformance(model, X_train_res, X_test_norm, y_train_res, y_test,
                                                target_names, target_original_value, save_plot, model_name="lr")
            show_performances.run()
            st.download_button("Download model", data=model_data, file_name="lr-model.pkl")



