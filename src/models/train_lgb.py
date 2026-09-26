import os
import sys
import time
import json
import pyarrow.parquet as pq
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class LGBMMatcherTrainer:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(base_dir, "configs", "model_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.lgb_params = self.config.get("lightgbm", {})
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

        # Build float32 X_train directly column-by-column to avoid pandas float64 36GB allocation
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
        best_iter = model.best_iteration if hasattr(model, "best_iteration") and model.best_iteration else model.n_estimators

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

            # Extract float32 batch matrix
            X_b = np.empty((b_len, len(feature_cols)), dtype=np.float32)
            for j, c in enumerate(feature_cols):
                X_b[:, j] = batch.column(c).to_numpy().astype(np.float32)

            b_probs = model.predict(X_b, num_iteration=best_iter)
            val_probs[offset:offset+b_len] = b_probs

            offset += b_len
            print(f"Predicted probabilities for {offset:,} / {total_rows:,} validation pairs...", flush=True)

        print(f"Validation inference complete in {time.time()-t0:.2f}s.", flush=True)
        return val_probs, val_labels, val_s1, val_cand, val_src, val_extra

    def train_baseline(self, feature_subset=None):
        train_path = os.path.join(base_dir, "artifacts", "features", "train_features.parquet")
        val_path = os.path.join(base_dir, "artifacts", "features", "validation_features.parquet")

        X_train, y_train, feature_names = self.load_train_matrix(train_path, feature_subset=feature_subset)

        # Sample 500,000 rows from training set as validation sample for LightGBM early stopping
        val_sample_size = min(500000, len(X_train))
        sample_indices = np.random.RandomState(self.seed).choice(len(X_train), size=val_sample_size, replace=False)
        X_val_sample = X_train[sample_indices]
        y_val_sample = y_train[sample_indices]

        print("\nConstructing LightGBM Datasets...", flush=True)
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
        dval_sample = lgb.Dataset(X_val_sample, label=y_val_sample, feature_name=feature_names, reference=dtrain, free_raw_data=False)

        params = {
            "objective": self.lgb_params.get("objective", "binary"),
            "metric": self.lgb_params.get("metric", "binary_logloss"),
            "boosting_type": self.lgb_params.get("boosting_type", "gbdt"),
            "learning_rate": self.lgb_params.get("learning_rate", 0.05),
            "num_leaves": self.lgb_params.get("num_leaves", 63),
            "max_depth": self.lgb_params.get("max_depth", -1),
            "min_child_samples": self.lgb_params.get("min_child_samples", 100),
            "subsample": self.lgb_params.get("subsample", 0.8),
            "colsample_bytree": self.lgb_params.get("colsample_bytree", 0.8),
            "reg_alpha": self.lgb_params.get("reg_alpha", 0.1),
            "reg_lambda": self.lgb_params.get("reg_lambda", 1.0),
            "random_state": self.seed,
            "n_jobs": -1,
            "verbose": -1
        }

        num_boost_round = self.lgb_params.get("n_estimators", 1000)
        early_stopping_rounds = self.lgb_params.get("early_stopping_rounds", 50)

        print("Training LightGBM baseline model...", flush=True)
        t0 = time.time()
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=True), lgb.log_evaluation(period=100)]
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval_sample],
            valid_names=["train", "val_sample"],
            callbacks=callbacks
        )
        train_time = time.time() - t0
        model_dir = os.path.join(base_dir, "artifacts", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "lightgbm_baseline.txt")
        model.save_model(model_path)
        print(f"Saved LightGBM model to {model_path}.", flush=True)

        # Free training matrices to release memory before streaming validation inference
        del dtrain, dval_sample, X_train, X_val_sample
        import gc
        gc.collect()

        # Stream Validation Predictions
        val_probs, val_labels, val_s1, val_cand, val_src, val_extra = self.stream_val_predictions(model, val_path, feature_names)

        # Compute Pairwise Metrics
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
            "best_iteration": model.best_iteration,
            "train_runtime_sec": train_time
        }

        return model, val_probs, val_s1, val_cand, val_src, val_extra, feature_names, pairwise_metrics

if __name__ == "__main__":
    trainer = LGBMMatcherTrainer()
    trainer.train_baseline()
