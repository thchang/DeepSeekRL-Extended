"""

"""

import os
import torch
import random
import textwrap
import numpy as np
from fpdf import FPDF
from typing import Any, Dict, Optional

import re

def clean_spaces_preserve_newlines(text):
    # Replace multiple spaces with a single space, but preserve newlines
    lines = text.split("\n")  # Split by newlines
    cleaned_lines = [" ".join(re.split(r"\s+", line)).strip() for line in lines]  # Remove extra spaces in each line
    return "\n".join(cleaned_lines)  # Join the lines back with newlines

# Example usage
pre_prompt = "  This    is  an   example   \n\n  Another   line   with  extra  spaces  "
cleaned_prompt = clean_spaces_preserve_newlines(pre_prompt)
print(cleaned_prompt)



def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across multiple libraries.
    
    This function sets consistent random seeds for Python's random module,
    NumPy, PyTorch (both CPU and CUDA), and configures CUDNN for deterministic
    operation. This ensures reproducible results across multiple runs.

    Args:
        seed: The random seed to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
