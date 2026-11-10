
from sklearn.linear_model import LogisticRegression

from common import run_comparison


if __name__ == "__main__":
    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
    )
    run_comparison("lr", model, scale_numeric=True)
