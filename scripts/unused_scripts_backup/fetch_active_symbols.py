#!/usr/bin/env python3
"""Fetch active symbols from database and write to stdout."""
import sys
import os
import re

# Redirect stderr to /dev/null
sys.stderr = open(os.devnull, 'w')

os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db'
sys.path.insert(0, '/opt/bist-pattern')

from app import create_app  # noqa: E402
from models import Stock  # noqa: E402

with create_app('default').app_context():
    symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).order_by(Stock.symbol).all()]

    # Denylist
    denylist = re.compile(r'USDTR|USDTRY|^XU|^OPX|^F_|VIOP|INDEX', re.IGNORECASE)
    symbols = [s for s in symbols if not denylist.search(s)]

    for s in symbols:
        print(s)
