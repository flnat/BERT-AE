import gc
import tempfile

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from pyod.models import iforest, knn, ecod, ocsvm, gmm
from sklearn import metrics, model_selection

from prepare_data import BagOfWords


def loop(model, param_grid: dict, x_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray,
         y_test: np.ndarray):
    param_grid = model_selection.ParameterGrid(param_grid)

    history = []

    for param in param_grid:
        estimator = model(**param)
        estimator.fit(x_train)

        prediction = estimator.decision_function(x_val)
        ap = metrics.average_precision_score(y_val, prediction)

        history.append({"model": estimator, "params": param, "metric": ap})

    history = pd.DataFrame(history)

    best_model = history.iloc[history.metric.idxmax()]

    y_hat = best_model.model.predict(x_test)
    scores = best_model.model.decision_function(x_test)

    test_results = {
        "f1": metrics.f1_score(y_test, y_hat),
        "recall": metrics.recall_score(y_test, y_hat),
        "precision": metrics.precision_score(y_test, y_hat),
        "auroc": metrics.roc_auc_score(y_test, scores),
        "average_precision": metrics.average_precision_score(y_test, scores)
    }

    diagnostic_plots = {
        "pr_rc.png": metrics.PrecisionRecallDisplay.from_predictions(y_test, scores).figure_,
        "roc.png": metrics.RocCurveDisplay.from_predictions(y_test, scores).figure_
    }

    return best_model.model, best_model.params, test_results, diagnostic_plots


def plot_complete_plots(fitted_models: list, model_names: list[str], plot_metric, plot_name, reduced_metric,
                        x_test, y_test):
    # fig = plt.figure(figsize=(9, 4.8))
    fig = plt.figure()
    fig.set_dpi(1200)
    ax = fig.add_subplot(111)

    df = pd.DataFrame({"y_true": y_test.tolist()})

    for name, model in zip(model_names, fitted_models):
        preds = model.decision_function(x_test)
        df[name] = preds.tolist()
        if plot_name == "Precision-Recall Curve":

            precision, recall, _ = plot_metric(y_test, preds)
            score = reduced_metric(y_test, preds)

            ax.plot(recall, precision, label=f"{name}, AP: {score:.2f}")

            ax.set_xlabel("Recall (Positive Label: 1)")
            ax.set_ylabel("Precision (Positive Label: 1)")

        else:
            fpr, tpr, _ = plot_metric(y_test, preds)
            score = reduced_metric(y_test, preds)

            ax.plot(fpr, tpr, label=f"{name}, AUROC: {score:.2f}")

            ax.set_ylabel("True Positive Rate (Positive Label: 1)")
            ax.set_xlabel("False Positive Rate (Positive Label: 1)")

    # ax.set_title(f"{plot_name}")

    box = ax.get_position()

    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=False, ncol=2)
    plt.tight_layout()
    return fig, df


if __name__ == "__main__":
    mlflow.autolog(disable=True)

    data = BagOfWords(ngramm_range=None)
    data.prepare()
    grids = [
        {
            "kernel": ["rbf", "poly", "sigmoid"],
            "nu": [0.1, 0.2, 0.3, 0.4]

        },
        {
            "n_estimators": [100, 200, 300],
            "max_features": [0.1 * i for i in range(1, 11)]
        },
        {
            "method": ["largest", "mean", "median"],
            "n_neighbors": [5 * i for i in range(1, 5)],
        },
        {
            "n_jobs": [1]
        },
        {
            "n_components": [1, 2, 3, 4],

        }, {
            "kernel": ["cosine", "rbf"]
        }]
    models = [ocsvm.OCSVM, iforest.IForest, knn.KNN, ecod.ECOD, gmm.GMM]
    model_name = ["One-Class SVM", "IsolationForest", "KNN", "ECOD", "Gaussian Mixture"]
    fitted_models = []

    # Create new mlflow experiment to log the models, if not already exists
    if mlflow.get_experiment_by_name("hdfs_baselines") is None:
        experiment_id = mlflow.create_experiment("hdfs_baselines")
    else:
        experiment_id = mlflow.get_experiment_by_name("hdfs_baselines").experiment_id


    for name, model, grid in zip(model_name, models, grids):
        estim, params, results, plots = loop(model, grid, x_train=data.x_train, x_test=data.x_test,
                                             y_test=data.y_test, x_val=data.x_val, y_val=data.y_val)

        with mlflow.start_run(experiment_id=experiment_id, run_name=name):
            mlflow.log_metrics(results)
            mlflow.log_params(params)

            for name, plot in plots.items():
                mlflow.log_figure(plot, name)

            with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
                joblib.dump(estim, tmp)
                mlflow.log_artifact(tmp.name)

        fitted_models.append(estim)
    gc.collect()
    # Make unified roc & pr rc plots
    roc_plot, roc_df = plot_complete_plots(fitted_models, model_name, metrics.roc_curve, "ROC", metrics.roc_auc_score,
                                           data.x_test, data.y_test)
    pr_rc_plot, _ = plot_complete_plots(fitted_models, model_name, metrics.precision_recall_curve,
                                        "Precision-Recall Curve", metrics.average_precision_score, data.x_test,
                                        data.y_test)

    last_run = mlflow.last_active_run().info.run_id

    with mlflow.start_run(run_id=last_run):
        mlflow.log_figure(roc_plot, "unified_roc.pdf")
        mlflow.log_figure(pr_rc_plot, "unified_pr_rc.pdf")
