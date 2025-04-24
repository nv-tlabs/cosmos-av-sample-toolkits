# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import click

import attrs
import numpy as np
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast
import pickle
from copy import deepcopy
import pandas as pd

tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b")
encoder = T5EncoderModel.from_pretrained("google-t5/t5-11b")

encoder.to("cuda")
encoder.eval()
encoder.requires_grad_(False)

@attrs.define
class EncodedSample:
    encoded_text: np.ndarray
    length: int
    attn_mask: np.ndarray
    offset_mappings: np.ndarray

    def truncate(self) -> None:
        self.encoded_text = self.encoded_text[0 : self.length].astype(np.float32)
        self.attn_mask = self.attn_mask[0 : self.length].astype(np.int32)
        if self.offset_mappings is not None:
            self.offset_mappings = self.offset_mappings[0 : self.length].astype(np.int32)

_max_length = 512
_output_mapping = True

def _encode_for_batch(
    prompts: list[str],
    truncate: bool = True,
):
    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=_max_length,
        return_length=True,
        return_offsets_mapping=_output_mapping,
    )

    # We expect all the processing is done in GPU.
    input_ids = batch_encoding.input_ids.cuda()
    attn_mask = batch_encoding.attention_mask.cuda()
    if _output_mapping:
        offsets_mapping = batch_encoding["offset_mapping"]
        offsets_mapping = offsets_mapping.cpu().numpy()
    else:
        offsets_mapping = None

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)  # type: ignore
    encoded_text = outputs.last_hidden_state

    lengths = attn_mask.sum(dim=1).cpu()  # batch_encoding["lengths"] is not valid for T5TokenizerFast
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0

    encoded_text = encoded_text.cpu().numpy()
    attn_mask = attn_mask.cpu().numpy()

    encoded_text = encoded_text[:, :_max_length]
    attn_mask = attn_mask[:, :_max_length]

    out = []  # type: list[EncodedSample]
    for idx in range(encoded_text.shape[0]):
        if _output_mapping:
            offsets = offsets_mapping[idx]
        else:
            offsets = None

        out.append(EncodedSample(encoded_text[idx].astype(np.float32), lengths[idx], attn_mask[idx], offsets))
    if truncate:
        for x in out:
            x.truncate()
    return out

@click.command()
@click.option(
    "--caption_file",
    type=str,
    default='./assets/waymo_caption.csv',
    help="Path to the folder containing caption files.",
)
@click.option(
    "--save_embd_folder",
    type=str,
    default='/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_holodeck/cosmos-mads-dataset-av/{CAMERAS.index(cur_camera)}/txt_mount_qwen_embed_fp32',
    help="Path to the folder where the embeddings will be saved.",
)
@click.option(
    "--videos_folder",
    type=str,
    default='',
    help="Optional. Path to the exported videos. Only encode prompts for videos that are present",
)
def main(caption_file, save_embd_folder, videos_folder):
    """
    Main function to create T5 embeddings from caption files.
    """
    os.makedirs(save_embd_folder, exist_ok=True)

    # Load the caption files and create embeddings
    df = pd.read_csv(caption_file, header=None)
    key_list = df.iloc[0].tolist()
    caption_list = df.iloc[1].tolist()
    if videos_folder:
        video_paths = [os.path.join(videos_folder, f) for f in os.listdir(videos_folder) if f.endswith(".mp4")]
        video_id_to_name = {}
        for vpath in video_paths:
            video_id_to_name[os.path.basename(vpath).split(".")[0][:-2]] = vpath
    prompts = []
    all_keys = []
    caption = {}
    for key, cap in zip(key_list, caption_list):
        # check if already processed
        if os.path.exists(os.path.join(save_embd_folder, key + ".pkl")):
            continue
        if videos_folder:
            if not key in video_id_to_name:
                continue
        all_keys.append(key)
        prompts.append(cap)
        caption[all_keys[-1]] = prompts[-1]

    date_string = '2025-04-18'
    save_embd_key = 'waymo_post_training'

    batch_size = 100
    n_batch = len(prompts) // batch_size + 1

    embedding_template = {
        "key": save_embd_key,
        "ground_truth": {
            "embeddings": {"t5_xxl": None, "t5_xxl_fp8": None, "byt5_small": None, "byt5_small_fp8": None},
            "subtree_pair_info": None,
        },
        "ground_truth_headline": None,
    }

    meta_tamplate = {
        "master_id": "1620516981",
        "media_type": "Film",
        "caption": "Moonstone",
        "headline": "Moonstone",
        "aiml_eligible": False,
        "nudity_filter": False,
        "footage_size": "hd15",
        "has_people": False,
        "create_date": date_string,
        "submit_date": date_string,
        "associated_keywords": "",
    }

    for i in tqdm(range(n_batch)):
        prompt = prompts[i * batch_size : min((i + 1) * batch_size, len(prompts))]
        out = _encode_for_batch(prompt)
        for j in range(len(out)):
            emb = out[j]
            p = prompt[j]
            curr_embedding_template = deepcopy(embedding_template)
            curr_embedding_template["ground_truth"]["embeddings"]["t5_xxl"] = emb.encoded_text
            curr_meta = deepcopy(meta_tamplate)
            curr_meta["caption"] = p
            k = all_keys[i * batch_size + j]
            assert p == caption[k]
            save_out = {"json": curr_meta, "__key__": k, "pickle": curr_embedding_template}
            pickle.dump(save_out, open(os.path.join(save_embd_folder, str(k) + ".pkl"), "wb"))

if __name__ == "__main__":
    main()