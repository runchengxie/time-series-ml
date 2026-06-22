"""Configuration via dataclass — no global mutable state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    """Immutable settings for a single experiment run.

    All values have sensible defaults and can be overridden via CLI.
    """

    # --- Data ---
    symbol: str = "000001.SZ"               # default: 平安银行
    symbols: list[str] = field(default_factory=lambda: ["000001.SZ"])
    start_date: str = "20150101"            # YYYYMMDD
    end_date: str = ""                      # YYYYMMDD; empty → computed
    data_lake_root: Path = Path(
        "/home/richard/data/market-data-platform/assets/tushare/"
        "a_share/daily/a_share_all_20150101_20260622_shadow_daily_clean/data"
    )
    instruments_path: Path = Path(
        "/home/richard/data/market-data-platform/assets/tushare/"
        "a_share/instruments/a_share_all_instruments_latest.parquet"
    )
    join_industry: bool = True             # auto-join 申万行业分类 from instruments
    neutralize_industry: bool = False       # cross-sectional industry neutralization
    min_listed_days: int = 252              # exclude stocks with < 1 year of data

    # --- Cache ---
    cache_dir: Path = Path(".cache")

    # --- Model ---
    test_size: float = 0.2
    up_threshold: float = 0.002             # +0.2% next-day return
    random_state: int = 42

    # --- Calibration ---
    calibrate: bool = False                 # fit CalibratedClassifierCV on train
    cv_method: str = "isotonic"            # "isotonic" | "sigmoid"

    # --- Signal filter ---
    prob_threshold: float = 0.50            # min predicted prob to trade

    # --- XGBoost hyper-params ---
    xgb_params: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "learning_rate": 0.01,
        "max_depth": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
    })

    # --- Cross-validation ---
    cv_splits: int = 5

    def __post_init__(self) -> None:
        if not self.end_date:
            from datetime import datetime
            object.__setattr__(self, "end_date", datetime.now().strftime("%Y%m%d"))

    @property
    def cache_file(self) -> Path:
        """Parameterised cache path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{self.symbol}_{self.start_date}_{self.end_date}.parquet"
