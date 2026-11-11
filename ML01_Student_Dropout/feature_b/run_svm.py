from sklearn.svm import SVC

from common import run_comparison


if __name__ == "__main__":
    model = SVC(
        kernel="rbf",
        C=10,
        gamma=0.01,
        class_weight="balanced",
    )
    run_comparison("svm", model, scale_numeric=True)
