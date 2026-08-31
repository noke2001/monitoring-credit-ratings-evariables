"""Whole-month difference between two datetimes, shared by the plot scripts."""
import pandas as pd


def months_between(later, earlier) -> int:
    a, b = pd.Timestamp(later), pd.Timestamp(earlier)
    return (a.year - b.year) * 12 + (a.month - b.month)
