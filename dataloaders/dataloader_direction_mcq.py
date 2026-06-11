# dataloaders/dataloader_direction_mcq.py

import json
import os
import numpy as np
from torch.utils.data import Dataset
from dataloaders.rawvideo_util import RawVideoExtractor


DEFAULT_CLASS_TEXTS = [
    "the object moves left",
    "the object moves right",
    "the object moves up",
    "the object moves down",
    "the object moves up left",
    "the object moves up right",
    "the object moves down left",
    "the object moves down right",
    "the object does not move",
]


class DirectionMCQ_DataLoader(Dataset):
    def __init__(
        self,
        json_path,
        features_path,
        tokenizer,
        max_words=32,
        max_frames=12,
        feature_framerate=3,
        frame_order=0,
        slice_framepos=0,
        class_texts=None,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.features_path = features_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.class_texts = class_texts or DEFAULT_CLASS_TEXTS

        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate,
            size=224,
        )

        # class prompt는 모든 샘플에서 같으니까 미리 만들어둠
        ids, masks, segs = [], [], []
        for text in self.class_texts:
            input_id, input_mask, segment_id = self._get_text(text)
            ids.append(input_id)
            masks.append(input_mask)
            segs.append(segment_id)

        self.class_input_ids = np.stack(ids, axis=0)       # [9, max_words]
        self.class_input_mask = np.stack(masks, axis=0)    # [9, max_words]
        self.class_segment_ids = np.stack(segs, axis=0)    # [9, max_words]

    def __len__(self):
        return len(self.data)

    def _get_text(self, text):
        words = self.tokenizer.tokenize(text)
        words = ["<|startoftext|>"] + words[: self.max_words - 2] + ["<|endoftext|>"]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        return (
            np.array(input_ids, dtype=np.int64),
            np.array(input_mask, dtype=np.int64),
            np.array(segment_ids, dtype=np.int64),
        )

    def _get_video(self, video_path):
        # video_path가 absolute면 os.path.join은 video_path를 그대로 씀
        full_path = os.path.join(self.features_path, video_path)

        video, video_mask = self.rawVideoExtractor.get_video_data(
            full_path,
            start_time=None,
            end_time=None,
            max_frames=self.max_frames,
        )

        return video, video_mask

    def __getitem__(self, idx):
        item = self.data[idx]

        video, video_mask = self._get_video(item["video_path"])
        label = int(item["label"])

        sample_id = str(item.get("sample_id", item.get("id", idx)))
        qa_type = str(item.get("qa_type", "direction_mcq"))
        question = str(item.get("question", "What is the motion direction?"))
        answer_text = str(
            item.get(
                "answer_text",
                item.get("direction_sentence", self.class_texts[label])
            )
        )
        debug_video_path = str(item["video_path"])

        return (
            self.class_input_ids.copy(),
            self.class_input_mask.copy(),
            self.class_segment_ids.copy(),
            video,
            video_mask,
            label,
            sample_id,
            qa_type,
            answer_text,
            question,
            debug_video_path,
        )