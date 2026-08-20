"""
Text branch for ReviewGuard.

Two interchangeable encoders, both operating on the real review text:

  TfidfSvdEncoder    - word (1-2gram) + character (3-5gram) TF-IDF reduced to a
                       dense latent space by truncated SVD (LSA), concatenated
                       with stylometric features in the tradition of Ott et al.
                       Runs anywhere; this is what the reported results use.

  RobertaEncoder     - fine-tuned roberta-base [CLS] embeddings. Implemented and
                       importable, but NOT used for the reported results: the
                       execution environment for this study had no GPU and no
                       route to the model hub, so it could not be run. It is
                       kept here so the transformer variant can be reproduced on
                       a machine that can reach huggingface.co.

Every encoder is fit on training data only and then applied to held-out data,
so the vocabulary and latent basis never see the evaluation fold.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

STYLOMETRIC_FEATURES = [
    "sty_char_count",
    "sty_word_count",
    "sty_mean_word_len",
    "sty_caps_ratio",
    "sty_exclaim_ratio",
    "sty_question_ratio",
    "sty_digit_ratio",
    "sty_punct_ratio",
    "sty_first_person_ratio",
    "sty_type_token_ratio",
]

_FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
_WORD_RE = re.compile(r"[a-z']+")


def stylometric_features(texts: pd.Series) -> pd.DataFrame:
    """Surface-form features that survive paraphrase but not rewriting."""
    out = {}
    s = texts.astype(str)
    chars = s.str.len().clip(lower=1)
    tokens = s.str.lower().map(lambda t: _WORD_RE.findall(t))
    n_words = tokens.map(len).clip(lower=1)

    out["sty_char_count"] = chars.astype(float)
    out["sty_word_count"] = n_words.astype(float)
    out["sty_mean_word_len"] = tokens.map(
        lambda ws: float(np.mean([len(w) for w in ws])) if ws else 0.0
    )
    out["sty_caps_ratio"] = s.str.count(r"[A-Z]") / chars
    out["sty_exclaim_ratio"] = s.str.count(r"!") / chars
    out["sty_question_ratio"] = s.str.count(r"\?") / chars
    out["sty_digit_ratio"] = s.str.count(r"[0-9]") / chars
    out["sty_punct_ratio"] = s.str.count(r"[^\w\s]") / chars
    out["sty_first_person_ratio"] = [
        sum(w in _FIRST_PERSON for w in ws) / max(len(ws), 1) for ws in tokens
    ]
    out["sty_type_token_ratio"] = [
        len(set(ws)) / max(len(ws), 1) for ws in tokens
    ]
    df = pd.DataFrame(out, index=texts.index)[STYLOMETRIC_FEATURES]
    return df.astype(float)


@dataclass
class TfidfSvdEncoder:
    """Sparse lexical features -> dense latent text representation.

    n_components is the dimensionality of the dense text vector handed to the
    fusion model. Stylometric features are appended to that vector.
    """

    n_components: int = 256
    word_max_features: int = 50_000
    char_max_features: int = 50_000
    min_df: int = 5
    random_state: int = 42
    include_stylometric: bool = True

    def __post_init__(self) -> None:
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_features=self.word_max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=self.min_df,
            max_features=self.char_max_features,
            sublinear_tf=True,
            lowercase=True,
        )
        self.svd = TruncatedSVD(
            n_components=self.n_components, random_state=self.random_state
        )
        self.scaler = StandardScaler()
        self.sty_scaler = StandardScaler()

    # -- sparse lexical matrix, used directly by the TF-IDF baselines ----------

    def fit_sparse(self, texts: pd.Series) -> sparse.csr_matrix:
        w = self.word_vec.fit_transform(texts)
        c = self.char_vec.fit_transform(texts)
        return sparse.hstack([w, c]).tocsr()

    def transform_sparse(self, texts: pd.Series) -> sparse.csr_matrix:
        w = self.word_vec.transform(texts)
        c = self.char_vec.transform(texts)
        return sparse.hstack([w, c]).tocsr()

    # -- dense representation, used by the MLP branches ------------------------

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        X = self.fit_sparse(texts)
        dense = self.scaler.fit_transform(self.svd.fit_transform(X))
        logger.info(
            "  TF-IDF %s -> SVD %d dims (%.1f%% variance retained)",
            X.shape,
            self.n_components,
            100 * self.svd.explained_variance_ratio_.sum(),
        )
        if not self.include_stylometric:
            return dense
        sty = self.sty_scaler.fit_transform(stylometric_features(texts).values)
        return np.hstack([dense, sty])

    def transform(self, texts: pd.Series) -> np.ndarray:
        X = self.transform_sparse(texts)
        dense = self.scaler.transform(self.svd.transform(X))
        if not self.include_stylometric:
            return dense
        sty = self.sty_scaler.transform(stylometric_features(texts).values)
        return np.hstack([dense, sty])

    @property
    def output_dim(self) -> int:
        extra = len(STYLOMETRIC_FEATURES) if self.include_stylometric else 0
        return self.n_components + extra

    def feature_names(self) -> List[str]:
        names = [f"svd_{i:03d}" for i in range(self.n_components)]
        if self.include_stylometric:
            names += list(STYLOMETRIC_FEATURES)
        return names


class RobertaEncoder:
    """Fine-tuned roberta-base [CLS] embeddings.

    Not exercised by the reported experiments - see module docstring. Requires
    `transformers`, network access to the model hub, and realistically a GPU
    (a single CPU epoch over 45,954 reviews at max_length=256 is hours).
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        max_length: int = 256,
        batch_size: int = 32,
        epochs: int = 2,
        lr: float = 2e-5,
        freeze_layers: int = 6,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.freeze_layers = freeze_layers
        self.device = device
        self._model = None
        self._tok = None

    def _build(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        for layer in self._model.encoder.layer[: self.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)

    def fit(self, texts: pd.Series, labels: np.ndarray) -> "RobertaEncoder":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from .losses import FocalLoss

        if self._model is None:
            self._build()
        enc = self._tok(
            list(texts), truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        head = nn.Linear(self._model.config.hidden_size, 1).to(self.device)
        ds = TensorDataset(
            enc["input_ids"], enc["attention_mask"], torch.FloatTensor(labels)
        )
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        params = [p for p in self._model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(list(head.parameters()) + params, lr=self.lr)
        crit = FocalLoss(alpha=float(1 - labels.mean()), gamma=2.0)

        self._model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for ids, mask, y in dl:
                ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)
                opt.zero_grad()
                cls = self._model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
                loss = crit(head(cls).squeeze(-1), y)
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
            logger.info("  RoBERTa epoch %d loss %.4f", epoch + 1, total / len(ds))
        return self

    @property
    def output_dim(self) -> int:
        return 768

    def transform(self, texts: pd.Series) -> np.ndarray:
        import torch

        if self._model is None:
            self._build()
        self._model.eval()
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = self._tok(
                batch, truncation=True, max_length=self.max_length,
                padding="max_length", return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                cls = self._model(**enc).last_hidden_state[:, 0]
            out.append(cls.cpu().numpy())
        return np.vstack(out)
