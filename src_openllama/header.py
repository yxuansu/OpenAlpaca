import torch
import datetime
import types
import gradio as gr
import mdtex2html
import pandas as pd
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
import transformers
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
from io import StringIO
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm
import os
import sys
import re
import math
from itertools import chain
import csv
import random
import json
import time
import pprint
import logging
from copy import deepcopy
import ipdb
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.activations import ACT2FN, get_activation
import pickle
import argparse
from torch.nn.utils.rnn import pad_sequence
import joblib
import torch.multiprocessing
from peft import LoraConfig, TaskType, get_peft_model
from utils import *

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
