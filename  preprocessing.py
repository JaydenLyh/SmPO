import io
from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm
from utils import  pickscore_utils

ps_selector = pickscore_utils.Selector('cuda')
dataset = load_from_disk('datasets/pickapic_v2')

def compute_scores(example):
    im_bytes_1 = example['jpg_0']
    im_bytes_2 = example['jpg_1']
    prompt = example['caption']

    im1 = Image.open(io.BytesIO(im_bytes_1)).convert("RGB")
    im2 = Image.open(io.BytesIO(im_bytes_2)).convert("RGB")
    ps_score = ps_selector.score(im1, prompt)
    ps_score_2 = ps_selector.score(im2, prompt)

    example['ps_score_0'] = ps_score
    example['ps_score_1'] = ps_score_2
    example['ps_label_0'] = 1 if ps_score > ps_score_2 else 0
    return example

dataset = dataset.map(compute_scores)
dataset.save_to_disk('datasets/pickapic_v2_scored')
