import os
import sys
import time
import json
import pyarrow.parquet as pq
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class XGBMatcherTrainer:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(base_dir, "configs", "model_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.xgb_params = self.config.get("xgboost", {})
        self.seed = self.config.get("seed", 42)

    def load_train_matrix(self, parquet_path, feature_subset=None):
        print(f"Loading training matrix from {parquet_path}...", flush=True)
        t0 = time.time()
        pf = pq.ParquetFile(parquet_path)
        all_cols = pf.schema.names

        meta_cols = {"source1_entity_id", "candidate_entity_id", "candidate_source", "label", "pair_id"}
        if feature_subset is not None:
            feature_cols = [c for c in feature_subset if c in all_cols and c not in meta_cols]
        else:
            feature_cols = [c for c in all_cols if c not in meta_cols]

        table = pf.read(columns=feature_cols + ["label"])
        y_train = table.column("label").to_numpy().astype(np.int8)

        num_rows = len(table)
        num_feats = len(feature_cols)
        X_train = np.empty((num_rows, num_feats), dtype=np.float32)

        for j, col_name in enumerate(feature_cols):
            X_train[:, j] = table.column(col_name).to_numpy().astype(np.float32)

        del table
        print(f"Loaded X_train ({num_rows:,} rows x {num_feats} float32 features, {X_train.nbytes/1e9:.2f} GB) in {time.time()-t0:.2f}s.", flush=True)
        return X_train, y_train, feature_cols

    def stream_val_predictions(self, model, val_parquet_path, feature_cols, chunk_size=2500000):
        print(f"Streaming validation predictions from {val_parquet_path} in chunks of {chunk_size:,} rows...", flush=True)
        t0 = time.time()
        pf = pq.ParquetFile(val_parquet_path)
        total_rows = pf.metadata.num_rows

        val_probs = np.empty(total_rows, dtype=np.float32)
        val_labels = np.empty(total_rows, dtype=np.int8)
        val_s1 = []
        val_cand = []
        val_src = []
        val_extra = {"name_address_contradiction_score": np.empty(total_rows, dtype=np.float32)}

        offset = 0
        best_iter = getattr(model, "best_iteration", None)

        for batch in pf.iter_batches(batch_size=chunk_size):
            b_len = len(batch)
            b_s1 = batch.column("source1_entity_id").to_pylist()
            b_cand = batch.column("candidate_entity_id").to_pylist()
            b_src = batch.column("candidate_source").to_pylist()
            b_lbl = batch.column("label").to_numpy().astype(np.int8)

            val_s1.extend(b_s1)
            val_cand.extend(b_cand)
            val_src.extend(b_src)
            val_labels[offset:offset+b_len] = b_lbl

            if "name_address_contradiction_score" in batch.schema.names:
                b_contra = batch.column("name_address_contradiction_score").to_numpy().astype(np.float32)
                val_extra["name_address_contradiction_score"][offset:offset+b_len] = b_contra

            X_b = np.empty((b_len, len(feature_cols)), dtype=np.float32)
            for j, c in enumerate(feature_cols):
                X_b[:, j] = batch.column(c).to_numpy().astype(np.float32)

            dval_b = xgb.DMatrix(X_b, feature_names=feature_cols)
            if best_iter is not None and best_iter > 0:
                b_probs = model.predict(dval_b, iteration_range=(0, best_iter + 1))
            else:
                b_probs = model.predict(dval_b)

            val_probs[offset:offset+b_len] = b_probs

            offset += b_len
            print(f"Predicted probabilities for {offset:,} / {total_rows:,} validation pairs...", flush=True)

        print(f"Validation inference complete in {time.time()-t0:.2f}s.", flush=True)
        return val_probs, val_labels, val_s1, val_cand, val_src, val_extra

    def train_baseline(self, feature_subset=None):
        train_path = os.path.join(base_dir, "artifacts", "features", "train_features.parquet")
        val_path = os.path.join(base_dir, "artifacts", "features", "validation_features.parquet")

        X_train, y_train, feature_names = self.load_train_matrix(train_path, feature_subset=feature_subset)

        # Sample 500,000 rows from training set for XGBoost early stopping validation
        val_sample_size = min(500000, len(X_train))
        sample_indices = np.random.RandomState(self.seed).choice(len(X_train), size=val_sample_size, replace=False)
        X_val_sample = X_train[sample_indices]
        y_val_sample = y_train[sample_indices]

        print("\nConstructing XGBoost DMatrix objects...", flush=True)
        dval_sample = xgb.DMatrix(X_val_sample, label=y_val_sample, feature_names=feature_names)
        del X_val_sample

        dtrain = xgb.QuantileDMatrix(X_train, label=y_train, feature_names=feature_names)
        del X_train
        import gc
        gc.collect()

        params = {
            "objective": self.xgb_params.get("objective", "binary:logistic"),
            "eval_metric": self.xgb_params.get("eval_metric", "logloss"),
            "learning_rate": self.xgb_params.get("learning_rate", 0.05),
            "max_depth": self.xgb_params.get("max_depth", 8),
            "min_child_weight": self.xgb_params.get("min_child_weight", 10),
            "subsample": self.xgb_params.get("subsample", 0.8),
            "colsample_bytree": self.xgb_params.get("colsample_bytree", 0.8),
            "alpha": self.xgb_params.get("reg_alpha", 0.1),
            "lambda": self.xgb_params.get("reg_lambda", 1.0),
            "tree_method": self.xgb_params.get("tree_method", "hist"),
            "seed": self.seed,
            "nthread": -1
        }

        num_boost_round = self.xgb_params.get("n_estimators", 1000)
        early_stopping_rounds = self.xgb_params.get("early_stopping_rounds", 50)

        print("Training XGBoost baseline model...", flush=True)
        t0 = time.time()
        
        evals = [(dtrain, "train"), (dval_sample, "val_sample")]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        train_time = time.time() - t0
        best_iter = getattr(model, "best_iteration", num_boost_round)
        print(f"XGBoost training complete in {train_time:.2f}s across {best_iter} iterations.", flush=True)

        model_dir = os.path.join(base_dir, "artifacts", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "xgboost_baseline.json")
        model.save_model(model_path)
        print(f"Saved XGBoost model to {model_path}.", flush=True)

        del dtrain, dval_sample
        gc.collect()

        # Stream Validation Predictions
        val_probs, val_labels, val_s1, val_cand, val_src, val_extra = self.stream_val_predictions(model, val_path, feature_names)

        # Calculate Pairwise Metrics
        val_loss = float(log_loss(val_labels, val_probs))
        val_auc = float(roc_auc_score(val_labels, val_probs))

        pred_labels = (val_probs >= 0.5).astype(int)
        tp = np.sum((pred_labels == 1) & (val_labels == 1))
        fp = np.sum((pred_labels == 1) & (val_labels == 0))
        fn = np.sum((pred_labels == 0) & (val_labels == 1))

        pair_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        pair_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        pairwise_metrics = {
            "val_logloss": val_loss,
            "val_auc": val_auc,
            "pair_precision": pair_prec,
            "pair_recall": pair_rec,
            "best_iteration": best_iter,
            "train_runtime_sec": train_time
        }

        return model, val_probs, val_s1, val_cand, val_src, val_extra, feature_names, pairwise_metrics

if __name__ == "__main__":
    trainer = XGBMatcherTrainer()
    trainer.train_baseline()

