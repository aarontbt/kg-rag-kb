#!/usr/bin/env python3

import sys
import os
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli.commands import main

if __name__ == "__main__":
    main()
