from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess(scale=False):
    numeric = [SimpleImputer(strategy="median")]
    if scale:
        numeric.append(StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer([
        ("numeric", make_pipeline(*numeric), make_column_selector(dtype_include="number")),
        ("categorical", categorical, make_column_selector(dtype_exclude="number")),
    ])


def add_features(frame):
    result = frame.copy()
    result["TotalSF"] = result["TotalBsmtSF"] + result["1stFlrSF"] + result["2ndFlrSF"]
    result["AgeAtSale"] = result["YrSold"] - result["YearBuilt"]
    result["YearsSinceRemodel"] = result["YrSold"] - result["YearRemodAdd"]
    return result
