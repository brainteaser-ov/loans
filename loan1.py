from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # older PyTorch
    from torch.cuda.amp import GradScaler, autocast  # type: ignore


PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
BYTE_OFFSET = 3
META_OFFSET = BYTE_OFFSET + 256

PROVENANCE_COLUMNS = [
    "ID",
    "Target_Form_ID",
    "Target_Form_Row_ID",
    "Target_WOLD_ID",
    "Target_Language_ID_CLDF",
    "Target_Glottocode",
    "Source_Language",
    "Source_Glottocode",
    "Parameter_ID",
    "Concepticon_ID",
    "Concepticon_Gloss",
    "Target_Borrowed",
    "Target_Borrowed_Score",
    "Borrowing_Comment",
    "Borrowing_Source",
]

FEATURE_NAMES = [
    "src_len",
    "tgt_len",
    "len_abs_diff",
    "len_ratio_short_long",
    "seqmatcher_ratio",
    "levenshtein_similarity",
    "common_prefix_ratio",
    "common_suffix_ratio",
    "bigram_jaccard",
    "trigram_jaccard",
    "first_char_same",
    "last_char_same",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value).strip())
    text = text.replace("’", "'").replace("ʻ", "'").replace("`", "'").replace("´", "'")
    return re.sub(r"\s+", " ", text).strip()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def char_ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def string_features(src: object, tgt: object) -> np.ndarray:
    a = normalize_text(src)
    b = normalize_text(tgt)
    la, lb = len(a), len(b)
    longer = max(1, la, lb)
    shorter = min(la, lb)

    seq_ratio = difflib.SequenceMatcher(None, a, b).ratio()
    lev_sim = 1.0 - (levenshtein_distance(a, b) / longer)

    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix += 1

    suffix = 0
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca != cb:
            break
        suffix += 1

    denom = max(1, shorter)
    bg_a, bg_b = char_ngrams(a, 2), char_ngrams(b, 2)
    tg_a, tg_b = char_ngrams(a, 3), char_ngrams(b, 3)
    bg_j = len(bg_a & bg_b) / max(1, len(bg_a | bg_b)) if (bg_a or bg_b) else 0.0
    tg_j = len(tg_a & tg_b) / max(1, len(tg_a | tg_b)) if (tg_a or tg_b) else 0.0

    return np.asarray(
        [
            la,
            lb,
            abs(la - lb),
            shorter / longer,
            seq_ratio,
            lev_sim,
            prefix / denom,
            suffix / denom,
            bg_j,
            tg_j,
            float(bool(a and b and a[0] == b[0])),
            float(bool(a and b and a[-1] == b[-1])),
        ],
        dtype=np.float32,
    )


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([string_features(s, t) for s, t in zip(df["src_form"], df["tgt_form"])]).astype(np.float32)


def read_excel_dataset(path: Path, sheet: str) -> pd.DataFrame:
    """Build the positive class exactly from WOLD rows with donor and target forms."""
    raw = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    needed = ["Target_Form", "Donor", "Target_Language_Name", "Meaning", "SemanticField", "SemanticCategory"]
    missing = [name for name in needed if name not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    df = raw.copy()
    df["src_form"] = df["Donor"].map(normalize_text)
    df["tgt_form"] = df["Target_Form"].map(normalize_text)
    df["tgt_lang"] = df["Target_Language_Name"].map(normalize_text)
    df["meaning"] = df["Meaning"].map(normalize_text)
    df["sem_field"] = df["SemanticField"].map(normalize_text)
    df["sem_cat"] = df["SemanticCategory"].map(normalize_text)

    mask = df["src_form"].ne("") & df["tgt_form"].ne("")
    pos = df.loc[mask].reset_index(drop=True)
    if pos.empty:
        raise ValueError("No positive pairs after filtering rows with donor and target forms.")

    keep = ["src_form", "tgt_form", "tgt_lang", "meaning", "sem_field", "sem_cat"]
    keep += [c for c in PROVENANCE_COLUMNS if c in pos.columns]
    pos = pos[keep].copy()
    pos["label_loan"] = 1
    pos["neg_type"] = "positive"
    pos["pair_source"] = "wold_positive"
    pos["src_form_norm"] = pos["src_form"].map(normalize_text)
    pos["tgt_form_norm"] = pos["tgt_form"].map(normalize_text)
    pos["row_uid"] = [f"pos_{i:07d}" for i in range(len(pos))]
    return pos

def parse_pair_file(path: Path, expected_label: Optional[int] = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return pd.DataFrame(columns=["form1", "form2", "label"])

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip().replace("\ufeff", "")
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            label = expected_label
            if parts[-1] in {"0", "1"}:
                label = int(parts[-1])
                parts = parts[:-1]
            if len(parts) == 2:
                form1, form2 = parts
            elif len(parts) >= 4:
                form1, form2 = parts[0], parts[2]
            else:
                form1, form2 = parts[0], parts[-1]
            rows.append({"form1": normalize_text(form1), "form2": normalize_text(form2), "label": label, "line_no": line_no})
    return pd.DataFrame(rows)


def sample_metadata(pos: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    base_cols = ["tgt_lang", "meaning", "sem_field", "sem_cat"]
    idx = [rng.randrange(len(pos)) for _ in range(n)]
    return pos.iloc[idx][base_cols].reset_index(drop=True)


def external_pair_negatives(pos: pd.DataFrame, path: Optional[Path], label: int, name: str, rng: random.Random) -> pd.DataFrame:
    if path is None or not path.exists():
        return empty_pairs()
    parsed = parse_pair_file(path, expected_label=label)
    parsed = parsed[parsed["label"].fillna(label).astype(int).eq(label)].copy()
    if parsed.empty:
        return empty_pairs()

    forward = parsed.rename(columns={"form1": "src_form", "form2": "tgt_form"})[["src_form", "tgt_form"]]
    backward = parsed.rename(columns={"form2": "src_form", "form1": "tgt_form"})[["src_form", "tgt_form"]]
    pairs = pd.concat([forward, backward], ignore_index=True).drop_duplicates()
    pairs = pairs[(pairs["src_form"] != "") & (pairs["tgt_form"] != "")].reset_index(drop=True)
    meta = sample_metadata(pos, len(pairs), rng)
    out = pd.concat([pairs, meta], axis=1)
    out["label_loan"] = 0
    out["neg_type"] = name
    out["pair_source"] = str(path)
    out["row_uid"] = [f"{name}_{i:07d}" for i in range(len(out))]
    return finish_negative_table(out)


def empty_pairs() -> pd.DataFrame:
    cols = [
        "src_form",
        "tgt_form",
        "tgt_lang",
        "meaning",
        "sem_field",
        "sem_cat",
        "label_loan",
        "neg_type",
        "pair_source",
        "row_uid",
        "src_form_norm",
        "tgt_form_norm",
    ]
    return pd.DataFrame(columns=cols)


def finish_negative_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["src_form"] = out["src_form"].map(normalize_text)
    out["tgt_form"] = out["tgt_form"].map(normalize_text)
    out["src_form_norm"] = out["src_form"].map(normalize_text)
    out["tgt_form_norm"] = out["tgt_form"].map(normalize_text)
    return out[(out["src_form_norm"] != "") & (out["tgt_form_norm"] != "")].reset_index(drop=True)


def assemble_pairs(
    pos: pd.DataFrame,
    *,
    seed: int,
    negative_ratio: float,
    cognates_path: Path,
    noncognates_path: Path,
) -> pd.DataFrame:
    """Assemble the article-style 1:1 corpus from WOLD positives and external negatives."""
    rng = random.Random(seed)
    n_neg = int(round(len(pos) * negative_ratio))
    if n_neg <= 0:
        raise ValueError("negative_ratio must produce at least one negative pair.")

    cognates = external_pair_negatives(pos, cognates_path, 1, "cognate_external", rng)
    noncognates = external_pair_negatives(pos, noncognates_path, 0, "noncognate_external", rng)
    if cognates.empty:
        raise ValueError(f"No cognate pairs were read from {cognates_path}.")
    if noncognates.empty:
        raise ValueError(f"No non-cognate pairs were read from {noncognates_path}.")

    neg_pool = pd.concat([cognates, noncognates], ignore_index=True)
    neg_pool = neg_pool.drop_duplicates(subset=["src_form_norm", "tgt_form_norm", "neg_type"]).reset_index(drop=True)
    if len(neg_pool) < n_neg:
        raise ValueError(
            f"Not enough external negative pairs after symmetrization and deduplication: "
            f"need {n_neg}, got {len(neg_pool)}. Add cognate/non-cognate pairs or lower --negative-ratio."
        )

    neg = neg_pool.sample(n=n_neg, random_state=seed).reset_index(drop=True)
    data = pd.concat([pos.copy(), neg], ignore_index=True)
    data = data.drop_duplicates(subset=["src_form_norm", "tgt_form_norm", "label_loan", "neg_type"])
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return data

def group_key(data: pd.DataFrame, split_mode: str) -> pd.Series:
    if split_mode == "target_form":
        return data["tgt_form_norm"].astype(str)
    if split_mode == "target_language":
        for col in ["Target_Language_ID_CLDF", "Target_Glottocode", "tgt_lang"]:
            if col in data.columns:
                return data[col].fillna("").astype(str)
    if split_mode == "language_pair":
        target = data.get("Target_Glottocode", data.get("tgt_lang", pd.Series([""] * len(data)))).fillna("").astype(str)
        source = data.get("Source_Glottocode", data.get("Source_Language", pd.Series([""] * len(data)))).fillna("").astype(str)
        return target + "|" + source
    raise ValueError(f"Unsupported split mode: {split_mode}")


def split_data(data: pd.DataFrame, seed: int, split_mode: str, test_size: float, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = group_key(data, split_mode)
    if groups.nunique() < 3:
        raise ValueError(f"Not enough groups for split mode {split_mode!r}.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(data, data["label_loan"], groups=groups))
    train_val = data.iloc[train_val_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    second_groups = group_key(train_val, split_mode)
    second_size = val_size / max(1e-9, 1.0 - test_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=second_size, random_state=seed + 1)
    train_idx, val_idx = next(splitter2.split(train_val, train_val["label_loan"], groups=second_groups))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test


def build_map(values: Iterable[object]) -> Dict[str, int]:
    items = sorted({normalize_text(v) for v in values if normalize_text(v)})
    return {"": 0, **{item: i + 1 for i, item in enumerate(items)}}


def to_byte_ids(text: object, max_bytes: int) -> List[int]:
    data = normalize_text(text).encode("utf-8", errors="ignore")[:max_bytes]
    return [BYTE_OFFSET + int(byte) for byte in data]


@dataclass
class EncodedPair:
    token_ids: List[int]
    segment_ids: List[int]
    features: np.ndarray
    label: int


class PairDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        maps: Dict[str, Dict[str, int]],
        bases: Dict[str, int],
        *,
        max_src_bytes: int,
        max_tgt_bytes: int,
        max_len: int,
        use_metadata: bool,
        use_features: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.maps = maps
        self.bases = bases
        self.max_src_bytes = max_src_bytes
        self.max_tgt_bytes = max_tgt_bytes
        self.max_len = max_len
        self.use_metadata = use_metadata
        self.use_features = use_features

    def __len__(self) -> int:
        return len(self.frame)

    def _metadata_tokens(self, row: pd.Series) -> List[int]:
        if not self.use_metadata:
            return []
        lang = self.maps["lang"].get(normalize_text(row.get("tgt_lang", "")), 0)
        field = self.maps["field"].get(normalize_text(row.get("sem_field", "")), 0)
        cat = self.maps["cat"].get(normalize_text(row.get("sem_cat", "")), 0)
        return [self.bases["lang"] + lang, self.bases["field"] + field, self.bases["cat"] + cat]

    def _encode(self, row: pd.Series) -> Tuple[List[int], List[int]]:
        tokens = [CLS_ID]
        segments = [0]
        meta = self._metadata_tokens(row)
        if meta:
            tokens.extend(meta)
            segments.extend([0] * len(meta))
        tokens.append(SEP_ID)
        segments.append(0)

        src = to_byte_ids(row["src_form"], self.max_src_bytes)
        tokens.extend(src + [SEP_ID])
        segments.extend([1] * (len(src) + 1))

        tgt = to_byte_ids(row["tgt_form"], self.max_tgt_bytes)
        tokens.extend(tgt + [SEP_ID])
        segments.extend([2] * (len(tgt) + 1))

        return tokens[: self.max_len], segments[: self.max_len]

    def __getitem__(self, index: int) -> EncodedPair:
        row = self.frame.iloc[index]
        tokens, segments = self._encode(row)
        feats = string_features(row["src_form"], row["tgt_form"]) if self.use_features else np.zeros(0, dtype=np.float32)
        return EncodedPair(tokens, segments, feats, int(row["label_loan"]))


def collate_pairs(max_len: int, feature_dim: int):
    def _collate(batch: Sequence[EncodedPair]):
        used_len = min(max(len(item.token_ids) for item in batch), max_len)
        token_ids = torch.full((len(batch), used_len), PAD_ID, dtype=torch.long)
        segment_ids = torch.zeros((len(batch), used_len), dtype=torch.long)
        features = torch.zeros((len(batch), feature_dim), dtype=torch.float32)
        labels = torch.zeros((len(batch),), dtype=torch.float32)

        for row_idx, item in enumerate(batch):
            toks = item.token_ids[:used_len]
            segs = item.segment_ids[:used_len]
            token_ids[row_idx, : len(toks)] = torch.tensor(toks, dtype=torch.long)
            segment_ids[row_idx, : len(segs)] = torch.tensor(segs, dtype=torch.long)
            if feature_dim:
                features[row_idx] = torch.tensor(item.features, dtype=torch.float32)
            labels[row_idx] = float(item.label)
        return token_ids, segment_ids, token_ids.eq(PAD_ID), features, labels

    return _collate


class ByteCrossEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_len: int,
        feature_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.segment_embedding = nn.Embedding(3, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model + feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, token_ids: torch.Tensor, segment_ids: torch.Tensor, pad_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0).expand(batch, length)
        hidden = self.token_embedding(token_ids) + self.segment_embedding(segment_ids) + self.position_embedding(positions)
        encoded = self.encoder(hidden, src_key_padding_mask=pad_mask)
        pooled = self.norm(encoded)[:, 0, :]
        if features.shape[1] > 0:
            pooled = torch.cat([pooled, features], dim=1)
        return self.classifier(pooled).squeeze(1)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    loss_fn = nn.BCEWithLogitsLoss()
    model.eval()
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    losses: list[float] = []
    counts: list[int] = []
    with torch.no_grad():
        for token_ids, segment_ids, pad_mask, features, labels in loader:
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            pad_mask = pad_mask.to(device)
            features = features.to(device)
            labels = labels.to(device)
            logits = model(token_ids, segment_ids, pad_mask, features)
            loss = loss_fn(logits, labels)
            losses.append(float(loss.item()))
            counts.append(labels.numel())
            labels_all.append(labels.detach().cpu().numpy())
            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
    y = np.concatenate(labels_all) if labels_all else np.zeros(0, dtype=np.float32)
    p = np.concatenate(probs_all) if probs_all else np.zeros(0, dtype=np.float32)
    metrics = binary_metrics(y.astype(int), p) if len(y) else {}
    metrics["loss"] = float(np.average(losses, weights=counts)) if losses else float("nan")
    return metrics, y.astype(int), p


def train_neural_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    maps: Dict[str, Dict[str, int]],
    bases: Dict[str, int],
    args: argparse.Namespace,
    outdir: Path,
    *,
    run_name: str,
    use_metadata: bool,
    use_features: bool,
) -> Dict[str, object]:
    feature_dim = len(FEATURE_NAMES) if use_features else 0
    max_len = 1 + (3 if use_metadata else 0) + 1 + args.max_src_bytes + 1 + args.max_tgt_bytes + 1
    vocab_size = bases["cat"] + len(maps["cat"])

    train_ds = PairDataset(
        train,
        maps,
        bases,
        max_src_bytes=args.max_src_bytes,
        max_tgt_bytes=args.max_tgt_bytes,
        max_len=max_len,
        use_metadata=use_metadata,
        use_features=use_features,
    )
    val_ds = PairDataset(
        val,
        maps,
        bases,
        max_src_bytes=args.max_src_bytes,
        max_tgt_bytes=args.max_tgt_bytes,
        max_len=max_len,
        use_metadata=use_metadata,
        use_features=use_features,
    )
    test_ds = PairDataset(
        test,
        maps,
        bases,
        max_src_bytes=args.max_src_bytes,
        max_tgt_bytes=args.max_tgt_bytes,
        max_len=max_len,
        use_metadata=use_metadata,
        use_features=use_features,
    )
    collate = collate_pairs(max_len, feature_dim)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ByteCrossEncoder(
        vocab_size=vocab_size,
        max_len=max_len,
        feature_dim=feature_dim,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    run_dir = outdir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for token_ids, segment_ids, pad_mask, features, labels in train_loader:
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            pad_mask = pad_mask.to(device)
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if sys.version_info >= (3, 0):
                try:
                    with autocast(device_type="cuda", enabled=bool(args.amp and device.type == "cuda")):
                        logits = model(token_ids, segment_ids, pad_mask, features)
                        loss = loss_fn(logits, labels)
                except TypeError:
                    with autocast(enabled=bool(args.amp and device.type == "cuda")):
                        logits = model(token_ids, segment_ids, pad_mask, features)
                        loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * labels.numel()
            total_count += labels.numel()

        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_count), **{f"val_{k}": v for k, v in val_metrics.items()}}
        metrics_rows.append(row)
        print(json.dumps({"run": run_name, **row}, ensure_ascii=False))

        if val_metrics.get("f1", -1.0) > best_f1 + 1e-5:
            best_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state": model.state_dict()}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    test_metrics, y_test, p_test = evaluate_model(model, test_loader, device)
    pd.DataFrame(metrics_rows).to_csv(run_dir / "training_metrics.csv", index=False)

    config = {
        "run_name": run_name,
        "use_metadata": use_metadata,
        "use_features": use_features,
        "feature_names": FEATURE_NAMES if use_features else [],
        "vocab_size": vocab_size,
        "max_len": max_len,
        "max_src_bytes": args.max_src_bytes,
        "max_tgt_bytes": args.max_tgt_bytes,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "ff_mult": args.ff_mult,
        "dropout": args.dropout,
        "best_epoch": best_epoch,
    }
    (run_dir / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metadata_maps.json").write_text(json.dumps({"maps": maps, "bases": bases}, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_table = test.copy()
    pred_table["prob_loan"] = p_test
    pred_table["pred_loan"] = (p_test >= 0.5).astype(int)
    pred_table["error_type"] = np.where(
        (pred_table["label_loan"] == 1) & (pred_table["pred_loan"] == 0),
        "false_negative",
        np.where((pred_table["label_loan"] == 0) & (pred_table["pred_loan"] == 1), "false_positive", "correct"),
    )
    save_error_analysis(pred_table, run_dir)

    return {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test_metrics": test_metrics,
        "model_parameters": int(sum(param.numel() for param in model.parameters())),
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    candidates = np.unique(np.round(scores, 6))
    if len(candidates) > 250:
        candidates = np.linspace(0.0, 1.0, 251)
    best_t = 0.5
    best_metrics = {"f1": -1.0}
    for threshold in candidates:
        metrics = binary_metrics(labels, scores, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_t = float(threshold)
            best_metrics = metrics
    return best_t, best_metrics


def run_baselines(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, outdir: Path) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    x_train = feature_matrix(train)
    x_val = feature_matrix(val)
    x_test = feature_matrix(test)
    y_train = train["label_loan"].astype(int).to_numpy()
    y_val = val["label_loan"].astype(int).to_numpy()
    y_test = test["label_loan"].astype(int).to_numpy()

    lev_col = FEATURE_NAMES.index("levenshtein_similarity")
    lev_threshold, lev_val = choose_threshold(y_val, x_val[:, lev_col])
    lev_test = binary_metrics(y_test, x_test[:, lev_col], lev_threshold)

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
    )
    logistic.fit(x_train, y_train)
    val_probs = logistic.predict_proba(x_val)[:, 1]
    log_threshold, log_val = choose_threshold(y_val, val_probs)
    test_probs = logistic.predict_proba(x_test)[:, 1]
    log_test = binary_metrics(y_test, test_probs, log_threshold)

    result = {
        "levenshtein_threshold": {"threshold": lev_threshold, "val": lev_val, "test": lev_test},
        "logistic_string_features": {"threshold": log_threshold, "val": log_val, "test": log_test, "feature_names": FEATURE_NAMES},
    }
    (outdir / "baseline_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def save_error_analysis(predictions: pd.DataFrame, outdir: Path) -> None:
    labels = predictions["label_loan"].astype(int).to_numpy()
    preds = predictions["pred_loan"].astype(int).to_numpy()
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    overall = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    overall.to_csv(outdir / "confusion_matrix_overall.csv")

    rows: list[dict[str, object]] = []
    for neg_type, chunk in predictions.groupby("neg_type", dropna=False):
        y = chunk["label_loan"].astype(int).to_numpy()
        p = chunk["pred_loan"].astype(int).to_numpy()
        cm_part = confusion_matrix(y, p, labels=[0, 1])
        rows.append(
            {
                "neg_type": neg_type,
                "n": len(chunk),
                "tn": int(cm_part[0, 0]),
                "fp": int(cm_part[0, 1]),
                "fn": int(cm_part[1, 0]),
                "tp": int(cm_part[1, 1]),
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "confusion_matrix_by_type.csv", index=False)

    predictions.to_csv(outdir / "test_predictions.csv", index=False)
    predictions[predictions["error_type"] == "false_positive"].sort_values("prob_loan", ascending=False).head(200).to_csv(
        outdir / "false_positives_top200.csv", index=False
    )
    predictions[predictions["error_type"] == "false_negative"].sort_values("prob_loan", ascending=True).head(200).to_csv(
        outdir / "false_negatives_top200.csv", index=False
    )


def save_run_manifest(args: argparse.Namespace, outdir: Path, data: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    manifest = {
        "input_data": str(args.data),
        "sheet": args.sheet,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "rows": {"all": len(data), "train": len(train), "val": len(val), "test": len(test)},
        "label_counts": data["label_loan"].value_counts(dropna=False).to_dict(),
        "neg_type_counts": data["neg_type"].value_counts(dropna=False).to_dict(),
        "used_training_columns": ["src_form", "tgt_form", "tgt_lang", "sem_field", "sem_cat"] + FEATURE_NAMES,
        "provenance_columns_kept_when_available": [c for c in PROVENANCE_COLUMNS if c in data.columns],
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_experiment(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]], Dict[str, int]]:
    positives = read_excel_dataset(args.data, args.sheet)
    if args.fast_dev_rows and args.fast_dev_rows > 0:
        keep_pos = max(20, args.fast_dev_rows // 2)
        positives = positives.sample(n=min(len(positives), keep_pos), random_state=args.seed).reset_index(drop=True)

    for path, label in [(args.cognates, "cognate"), (args.noncognates, "non-cognate")]:
        if path is None:
            raise ValueError(f"The {label} file is required for this experiment.")
        if not path.exists():
            raise FileNotFoundError(f"The {label} file was not found: {path}")

    data = assemble_pairs(
        positives,
        seed=args.seed,
        negative_ratio=args.negative_ratio,
        cognates_path=args.cognates,
        noncognates_path=args.noncognates,
    )
    if args.fast_dev_rows and args.fast_dev_rows > 0:
        data = data.sample(n=min(len(data), args.fast_dev_rows), random_state=args.seed).reset_index(drop=True)

    train, val, test = split_data(data, args.seed, args.split_mode, args.test_size, args.val_size)
    maps = {
        "lang": build_map(train["tgt_lang"]),
        "field": build_map(train["sem_field"]),
        "cat": build_map(train["sem_cat"]),
    }
    bases = {
        "lang": META_OFFSET,
        "field": META_OFFSET + len(maps["lang"]),
        "cat": META_OFFSET + len(maps["lang"]) + len(maps["field"]),
    }
    return data, train, val, test, maps, bases

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WOLD loanword-pair classifier with article-style negative sampling")
    parser.add_argument("--data", type=Path, default=Path("dataset.xlsx"), help="WOLD-derived Excel file")
    parser.add_argument("--sheet", default="dataset", help="Excel sheet with pair data")
    parser.add_argument("--outdir", type=Path, default=Path("loan_artifacts"), help="Output directory")
    parser.add_argument("--cognates", type=Path, required=True, help="Cognate pairs file; used as hard negatives")
    parser.add_argument("--noncognates", type=Path, required=True, help="Non-cognate pairs file; used as ordinary negatives")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Number of negatives per positive")
    parser.add_argument("--split-mode", choices=["target_form", "target_language", "language_pair"], default="target_form")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-dev-rows", type=int, default=0, help="Use a small sample for debugging")

    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--run-ablations", action="store_true")

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None, help="cpu, cuda, or empty for auto")
    parser.add_argument("--max-src-bytes", type=int, default=64)
    parser.add_argument("--max-tgt-bytes", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    data, train, val, test, maps, bases = prepare_experiment(args)
    data.to_csv(args.outdir / "prepared_pairs.csv", index=False)
    train.to_csv(args.outdir / "train_pairs.csv", index=False)
    val.to_csv(args.outdir / "val_pairs.csv", index=False)
    test.to_csv(args.outdir / "test_pairs.csv", index=False)
    save_run_manifest(args, args.outdir, data, train, val, test)

    results: dict[str, object] = {}
    if not args.skip_baselines:
        results["baselines"] = run_baselines(train, val, test, args.outdir / "baselines")

    neural_results: list[dict[str, object]] = []
    if not args.skip_neural:
        neural_results.append(
            train_neural_model(
                train,
                val,
                test,
                maps,
                bases,
                args,
                args.outdir,
                run_name="full",
                use_metadata=True,
                use_features=True,
            )
        )
        if args.run_ablations:
            neural_results.append(
                train_neural_model(train, val, test, maps, bases, args, args.outdir, run_name="no_metadata", use_metadata=False, use_features=True)
            )
            neural_results.append(
                train_neural_model(train, val, test, maps, bases, args, args.outdir, run_name="no_string_features", use_metadata=True, use_features=False)
            )
            neural_results.append(
                train_neural_model(train, val, test, maps, bases, args, args.outdir, run_name="bytes_only", use_metadata=False, use_features=False)
            )
        pd.DataFrame(neural_results).to_json(args.outdir / "neural_runs_summary.json", orient="records", force_ascii=False, indent=2)
        results["neural"] = neural_results

    (args.outdir / "results_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"outdir": str(args.outdir), "rows": len(data), "train": len(train), "val": len(val), "test": len(test)}, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
