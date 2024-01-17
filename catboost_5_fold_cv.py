import gc
import os
import time
import warnings
from itertools import combinations
from warnings import simplefilter
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
is_offline = False
is_train = True
is_infer = True
max_lookback = np.nan
split_day = 435

df = pd.read_csv("train.csv")
df = df.dropna(subset=["target"])
df.reset_index(drop=True, inplace=True)
df_shape = df.shape


def reduce_mem_usage(df, verbose=0):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024 ** 2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")
    return df


from numba import njit, prange


@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val

            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features


def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
    features_array = compute_triplet_imbalance(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features


def imbalance_features(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")

    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values

    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])

    # Calculate various statistical aggregation features
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)

    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)

    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum',
                'size_imbalance']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

    return df.replace([np.inf, -np.inf], 0)


def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60
    df["minute"] = df["seconds_in_bucket"] // 60
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df


def generate_all_features(df):
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]

    # Generate imbalance features
    df = imbalance_features(df)
    df = other_features(df)
    gc.collect()
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]

    return df[feature_name]

if is_offline:
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    print("Offline mode")
    print(f"train : {df_train.shape}, valid : {df_valid.shape}")
else:
    df_train = df
    print("Online mode")

if is_train:
    global_stock_id_feats = {
        "median_size": df_train.groupby("stock_id")["bid_size"].median() + df_train.groupby("stock_id")["ask_size"].median(),
        "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
        "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
        "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
    }
    if is_offline:
        df_train_feats = generate_all_features(df_train)
        print("Build Train Feats Finished.")
        df_valid_feats = generate_all_features(df_valid)
        print("Build Valid Feats Finished.")
        df_valid_feats = reduce_mem_usage(df_valid_feats)
    else:
        df_train_feats = generate_all_features(df_train)
        print(df_train_feats.shape)
        print("Build Online Train Feats Finished.")

    df_train_feats = reduce_mem_usage(df_train_feats)

import numpy as np
import catboost
from catboost import CatBoostRegressor
from catboost import EShapCalcType, EFeaturesSelectionAlgorithm
from catboost import Pool
from sklearn.metrics import mean_absolute_error
import gc

cat_params = dict(loss_function='MAE',
                  iterations=25000,
                  score_function='Cosine',
                  eval_metric='MAE',
                  metric_period=100,
                  task_type='GPU',
                  od_type='Iter',
                  od_wait=100,
                  verbose=200,
                  random_seed=7,
                  border_count=32,
                  use_best_model=True,
                  bootstrap_type='Bernoulli',
                  subsample=0.60,
                  random_strength=3,  # more bagging to reduce overfitting
                  depth=12,
                  grow_policy='Lossguide',
                  l2_leaf_reg=8,
                  max_leaves = 512,
                  nan_mode='Min',
                  learning_rate=0.00877,
                  )
feature_name = list(df_train_feats.columns)
print(f"Feature length = {len(feature_name)}")

num_folds = 5
fold_size = 480 // num_folds
gap = 5

models = []
scores = []

model_save_path = 'modelitos_para_cat'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

date_ids = df_train['date_id'].values

for i in range(num_folds):
    start = i * fold_size
    end = start + fold_size
    if i < num_folds - 1:  # No need to purge after the last fold
        purged_start = end - 2
        purged_end = end + gap + 2
        train_indices = (date_ids >= start) & (date_ids < purged_start) | (date_ids > purged_end)
    else:
        train_indices = (date_ids >= start) & (date_ids < end)

    test_indices = (date_ids >= end) & (date_ids < end + fold_size)

    df_fold_train = df_train_feats[train_indices]
    df_fold_train_target = df_train['target'][train_indices]
    df_fold_valid = df_train_feats[test_indices]
    df_fold_valid_target = df_train['target'][test_indices]

    print(f"Fold {i + 1} Model Training")

    # Train a LightGBM model for the current fold
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        df_fold_train[feature_name],
        df_fold_train_target,
        eval_set=[(df_fold_valid[feature_name], df_fold_valid_target)],
    )

    models.append(cat_model)
    # Save the model to a file
    model_filename = os.path.join(model_save_path, f'doblez_{i + 1}.txt')
    cat_model.save_model(model_filename)
    print(f"Model for fold {i + 1} saved to {model_filename}")

    # Evaluate model performance on the validation set
    fold_predictions = cat_model.predict(df_fold_valid[feature_name])
    fold_score = mean_absolute_error(fold_predictions, df_fold_valid_target)
    scores.append(fold_score)
    print(f"Fold {i + 1} MAE: {fold_score}")

    # Free up memory by deleting fold specific variables
    del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
    gc.collect()

# Calculate the average best iteration from all regular folds
average_best_iteration = int(np.mean([model.get_best_iteration() for model in models]))

# Update the lgb_params with the average best iteration
final_model_params = cat_params.copy()
final_model_params['iterations'] = average_best_iteration
final_model_params['use_best_model'] = False

print(f"Training final model with average best iteration: {average_best_iteration}")

# Train the final model on the entire dataset
final_model = CatBoostRegressor(**final_model_params)
final_model.fit(
    df_train_feats[feature_name],
    df_train['target'],
)

# Append the final model to the list of models
models.append(final_model)

# Save the final model to a file
final_model_filename = os.path.join(model_save_path, 'doblez-conjunto.txt')
final_model.save_model(final_model_filename)
print(f"Final model saved to {final_model_filename}")

# Now 'models' holds the trained models for each fold and 'scores' holds the validation scores
print(f"Average MAE across all folds: {np.mean(scores)}")