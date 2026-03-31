"""data_processor.py — company data processing without PyTorch tensors."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import dask.dataframe as dd  # optional dependency
    _DASK_AVAILABLE = True
except ImportError:
    _DASK_AVAILABLE = False


class CompanyDataProcessor:
    """Processes company data for AI consulting analysis."""

    _REQUIRED_FIELDS = {
        "financial": ["revenue", "costs", "profit_margins", "growth_rate"],
        "operational": ["employees", "locations", "capacity_utilization"],
        "market": ["market_share", "competitors", "target_segments"],
        "strategic": ["current_initiatives", "challenges", "objectives"],
    }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_data(self, data: Union[Dict, List[Dict]]) -> bool:
        if isinstance(data, dict) and "datasets" in data:
            return all(self._validate_single(ds) for ds in data["datasets"])
        return self._validate_single(data)

    def _validate_single(self, data: Dict) -> bool:
        try:
            for category, fields in self._REQUIRED_FIELDS.items():
                if category not in data:
                    logger.warning("Missing category: %s", category)
                    return False
                for f in fields:
                    if f not in data[category]:
                        logger.warning("Missing field %s in %s", f, category)
                        return False
            return True
        except Exception as exc:
            logger.error("Validation error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Processing — returns plain dicts instead of torch.Tensor
    # ------------------------------------------------------------------

    def process_financial_data(self, financial_data: Dict) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for f in self._REQUIRED_FIELDS["financial"]:
            value = financial_data.get(f, 0)
            if isinstance(value, str):
                try:
                    value = float(value.replace(",", "").replace("$", ""))
                except ValueError:
                    value = 0.0
            result[f] = float(value)
        return result

    def process_market_data(self, market_data: Dict) -> Dict[str, float]:
        share_raw = str(market_data.get("market_share", "0")).rstrip("%")
        try:
            market_share = float(share_raw) / 100
        except ValueError:
            market_share = 0.0

        return {
            "market_share": market_share,
            "competitor_count": float(len(market_data.get("competitors", []))),
            "segment_count": float(len(market_data.get("target_segments", []))),
        }

    def process_company_data(self, company_data: Dict) -> Dict[str, Dict]:
        return {
            "financial": self.process_financial_data(company_data.get("financial", {})),
            "market": self.process_market_data(company_data.get("market", {})),
        }

    def process_strategic_data(self, strategic_data: Dict) -> Dict[str, int]:
        return {
            "initiative_count": len(strategic_data.get("current_initiatives", [])),
            "challenge_count": len(strategic_data.get("challenges", [])),
            "objective_count": len(strategic_data.get("objectives", [])),
        }

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        path = Path(file_path)
        if path.suffix.lower() == ".csv":
            if _DASK_AVAILABLE:
                return dd.read_csv(str(path)).compute()
            return pd.read_csv(path)
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": list(data.shape),
            "columns": list(data.columns),
            "summary": data.describe(include="all").to_dict(),
            "null_counts": data.isnull().sum().to_dict(),
        }

    def save_results(self, results: Any, output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        logger.info("Results saved to %s", path)

    def generate_visualization(self, results: Any, output_path: Union[str, Path]) -> None:
        """Write a minimal HTML report (no Plotly dependency required at this layer)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = (
            "<html><body><h1>AIC Analysis Report</h1>"
            f"<pre>{json.dumps(results, indent=2, default=str)}</pre>"
            "</body></html>"
        )
        path.write_text(html, encoding="utf-8")
        logger.info("Visualization saved to %s", path)
