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
    final_oos_size: float = 0.0            # 0 = disabled; 0.10 = last 10% as final OOS holdout
    up_threshold: float = 0.002             # +0.2% next-day return (legacy binary)
    random_state: int = 42

    # --- Triple Barrier Labeling ---
    triple_barrier: bool = False            # True = use triple barrier labels (-1/0/+1)
    holding_period: int = 5                 # days to hold (time barrier)
    profit_take: float = 0.05               # profit barrier as fraction (e.g. 0.05 = 5%)
    stop_loss: float = 0.03                 # stop barrier as fraction (e.g. 0.03 = 3%)

    # --- Calibration ---
    calibrate: bool = False                 # fit CalibratedClassifierCV on train
    cv_method: str = "isotonic"            # "isotonic" | "sigmoid"

    # --- Signal filter ---
    prob_threshold: float = 0.50            # min predicted prob to trade

    # --- Sample weighting ---
    sample_weight_halflife: int = 0         # 0 = disabled (equal weight); >0 = exp_decay
                                            # halflife in trading days (e.g. 252 = 1 year)

    # --- Training window ---
    train_window_days: int = 0              # 0 = use all history; >0 = rolling window
                                            # (e.g. 504 = ~2 years of trading days)

    # --- Regression mode ---
    regression: bool = False                # True = predict future_return (regression)
                                            # instead of binary up/down (classification)

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

    # --- Backtest ---
    backtest: bool = False
    backtest_cost_bps: float = 5.0          # legacy: round-trip cost (deprecated by buy/sell split)
    backtest_buy_cost_bps: float = 0.0      # buy-side cost in bps (A-share: 0 for stamp duty)
    backtest_sell_cost_bps: float = 0.0     # sell-side cost in bps; if 0, uses backtest_cost_bps/2
    backtest_enforce_price_limit: bool = True  # skip signals when stock hits daily price limit

    # --- Liquidity filter ---
    min_daily_amount: float = 0.0           # 0 = disabled; >0 = filter out dates below this
                                            # daily turnover in CNY (e.g. 1_000_000 = 100万)

    # --- Ablation ---
    ablation: bool = False

    def __post_init__(self) -> None:
        if not self.end_date:
            from datetime import datetime
            object.__setattr__(self, "end_date", datetime.now().strftime("%Y%m%d"))

        # Derive sell cost from legacy backtest_cost_bps if not explicitly set
        if self.backtest_sell_cost_bps == 0.0 and self.backtest_cost_bps > 0:
            object.__setattr__(self, "backtest_sell_cost_bps", self.backtest_cost_bps / 2.0)

    @property
    def cache_file(self) -> Path:
        """Parameterised cache path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{self.symbol}_{self.start_date}_{self.end_date}.parquet"
