#!/usr/bin/env python3
"""
AlphaIgnitor3 - Portal Generator CLI Wrapper
Delegates to alphaignitor.pipeline.portal.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphaignitor.pipeline.portal import main

if __name__ == "__main__":
    main()
