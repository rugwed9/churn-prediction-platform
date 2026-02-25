"""Synthetic SaaS user behavior data generator.

Generates realistic churn data for 10,000 users over 12 months with
correlated features that mimic real-world SaaS usage patterns.

Key design decisions:
- Features are correlated (e.g., low login frequency correlates with
  higher churn probability) using a latent "engagement" variable
- Temporal patterns: users who will churn show declining engagement
  over their final months
- Class imbalance matches typical SaaS churn (~18%)
- Monthly snapshots enable point-in-time feature engineering
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DataConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


class ChurnDataGenerator:
    """Generates synthetic SaaS user behavior data with realistic patterns."""

    PLAN_TIERS = ["free", "starter", "professional", "enterprise"]
    PLAN_WEIGHTS = [0.30, 0.35, 0.25, 0.10]
    PLAN_PRICES = {"free": 0, "starter": 29, "professional": 99, "enterprise": 299}

    BILLING_CYCLES = ["monthly", "annual"]
    BILLING_WEIGHTS = [0.65, 0.35]

    SIGNUP_CHANNELS = ["organic", "paid_search", "referral", "partner"]
    SIGNUP_WEIGHTS = [0.40, 0.25, 0.20, 0.15]

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> pd.DataFrame:
        """Generate the full dataset with user-month observations.

        Returns:
            DataFrame with one row per user containing aggregated features
            and a binary churn label.
        """
        logger.info(
            "Generating data: %d users, %d months, %.0f%% churn rate",
            self.config.n_users, self.config.n_months, self.config.churn_rate * 100,
        )

        users = self._generate_user_profiles()
        monthly = self._generate_monthly_behavior(users)
        final = self._aggregate_features(users, monthly)

        logger.info(
            "Generated dataset: %d rows, %d features, %.1f%% churn",
            len(final), len(final.columns), final["churned"].mean() * 100,
        )
        return final

    def _generate_user_profiles(self) -> pd.DataFrame:
        """Create static user-level attributes."""
        n = self.config.n_users

        # Latent engagement score (drives all behavioral features)
        engagement = self.rng.beta(2, 2, size=n)

        # Churn probability inversely correlated with engagement
        churn_logit = -3.0 + 5.0 * (1 - engagement) + self.rng.normal(0, 0.5, n)
        churn_prob = 1 / (1 + np.exp(-churn_logit))

        # Calibrate to target churn rate
        threshold = np.percentile(churn_prob, (1 - self.config.churn_rate) * 100)
        churned = (churn_prob >= threshold).astype(int)

        # Plan tier (higher engagement -> higher tier)
        plan_probs = np.array(self.PLAN_WEIGHTS)
        plan_tier = []
        for eng in engagement:
            adjusted = plan_probs.copy()
            adjusted[2:] *= (1 + eng)  # Boost pro/enterprise for engaged users
            adjusted /= adjusted.sum()
            plan_tier.append(self.rng.choice(self.PLAN_TIERS, p=adjusted))

        # Signup date spread over first 6 months
        base_date = datetime(2023, 1, 1)
        signup_days = self.rng.integers(0, 180, size=n)
        signup_dates = [base_date + timedelta(days=int(d)) for d in signup_days]

        # Churn month for churned users (biased toward later months)
        churn_month = np.zeros(n, dtype=int)
        churned_mask = churned == 1
        n_churned = churned_mask.sum()
        churn_month[churned_mask] = self.rng.integers(
            4, self.config.n_months + 1, size=n_churned
        )

        return pd.DataFrame({
            "user_id": [f"user_{i:05d}" for i in range(n)],
            "engagement_latent": engagement,
            "churned": churned,
            "churn_month": churn_month,
            "plan_tier": plan_tier,
            "billing_cycle": self.rng.choice(
                self.BILLING_CYCLES, size=n, p=self.BILLING_WEIGHTS
            ),
            "signup_channel": self.rng.choice(
                self.SIGNUP_CHANNELS, size=n, p=self.SIGNUP_WEIGHTS
            ),
            "signup_date": signup_dates,
            "company_size": self.rng.choice(
                ["1-10", "11-50", "51-200", "201-1000", "1000+"],
                size=n,
                p=[0.30, 0.30, 0.20, 0.15, 0.05],
            ),
        })

    def _generate_monthly_behavior(self, users: pd.DataFrame) -> pd.DataFrame:
        """Generate monthly behavioral features for each user.

        Users who will churn show declining engagement in their final months.
        """
        records = []
        n_months = self.config.n_months

        for _, user in users.iterrows():
            eng = user["engagement_latent"]
            will_churn = user["churned"] == 1
            churn_m = user["churn_month"]

            for month in range(1, n_months + 1):
                # If user already churned, skip
                if will_churn and month > churn_m:
                    continue

                # Engagement decay for churning users approaching churn month
                if will_churn and churn_m > 0:
                    months_to_churn = churn_m - month
                    decay = max(0.1, months_to_churn / churn_m)
                    effective_eng = eng * decay
                else:
                    effective_eng = eng * (1 + 0.02 * month)  # Slight growth for retained

                effective_eng = np.clip(effective_eng, 0.01, 0.99)

                # Behavioral features driven by effective engagement
                login_freq = max(0, self.rng.poisson(effective_eng * 25))
                session_dur = max(0.5, self.rng.gamma(2, effective_eng * 15))
                feature_usage = np.clip(
                    effective_eng * 100 + self.rng.normal(0, 10), 0, 100
                )
                support_tickets = self.rng.poisson(max(0.1, (1 - effective_eng) * 3))
                active_days = min(30, max(0, self.rng.binomial(30, effective_eng)))
                pages_per_session = max(1, self.rng.poisson(effective_eng * 8))

                # Days since last login (inverse of engagement)
                days_since_login = max(0, self.rng.exponential(
                    max(0.5, (1 - effective_eng) * 15)
                ))

                records.append({
                    "user_id": user["user_id"],
                    "month": month,
                    "login_frequency": login_freq,
                    "avg_session_duration_min": round(session_dur, 1),
                    "feature_usage_score": round(feature_usage, 1),
                    "support_tickets": support_tickets,
                    "days_since_last_login": round(days_since_login, 1),
                    "monthly_active_days": active_days,
                    "pages_per_session": pages_per_session,
                })

        return pd.DataFrame(records)

    def _aggregate_features(
        self, users: pd.DataFrame, monthly: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate monthly data into user-level features.

        Point-in-time correct: for churned users, only uses data up to
        the month BEFORE churn. For retained users, uses all 12 months.
        """
        results = []

        for _, user in users.iterrows():
            uid = user["user_id"]
            user_monthly = monthly[monthly["user_id"] == uid].copy()

            if user["churned"] == 1 and user["churn_month"] > 0:
                # Only use data BEFORE the churn month (no lookahead)
                cutoff = user["churn_month"] - 1
                user_monthly = user_monthly[user_monthly["month"] <= cutoff]

            if user_monthly.empty:
                continue

            # Recent window (last 3 months of available data)
            max_month = user_monthly["month"].max()
            recent = user_monthly[user_monthly["month"] >= max_month - 2]

            # Trend features (recent vs. earlier)
            early = user_monthly[user_monthly["month"] <= max_month - 3]

            login_trend = 0.0
            if not early.empty and early["login_frequency"].mean() > 0:
                login_trend = (
                    recent["login_frequency"].mean() - early["login_frequency"].mean()
                ) / early["login_frequency"].mean()

            row = {
                "user_id": uid,
                "login_frequency": recent["login_frequency"].mean(),
                "avg_session_duration_min": recent["avg_session_duration_min"].mean(),
                "feature_usage_score": recent["feature_usage_score"].mean(),
                "support_tickets_total": user_monthly["support_tickets"].sum(),
                "support_tickets_recent": recent["support_tickets"].sum(),
                "days_since_last_login": recent["days_since_last_login"].iloc[-1],
                "monthly_active_days": recent["monthly_active_days"].mean(),
                "pages_per_session": recent["pages_per_session"].mean(),
                "login_frequency_trend": login_trend,
                "session_duration_std": user_monthly["avg_session_duration_min"].std(),
                "months_active": len(user_monthly),
                "plan_tier": user["plan_tier"],
                "billing_cycle": user["billing_cycle"],
                "signup_channel": user["signup_channel"],
                "company_size": user["company_size"],
                "mrr": self.PLAN_PRICES.get(user["plan_tier"], 0),
                "churned": user["churned"],
            }
            results.append(row)

        df = pd.DataFrame(results)

        # Fill NaN in std columns for users with single month
        df["session_duration_std"] = df["session_duration_std"].fillna(0)

        return df

    def save(self, df: pd.DataFrame, filename: str = "churn_dataset.parquet") -> Path:
        """Save dataset to parquet."""
        output_dir = PROJECT_ROOT / self.config.raw_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info("Saved dataset (%d rows) to %s", len(df), output_path)
        return output_path
