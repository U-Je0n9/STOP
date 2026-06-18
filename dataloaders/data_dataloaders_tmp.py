# coding=utf-8
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Sampler
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
from dataloaders.dataloader_activitynet_retrieval import ActivityNet_DataLoader
from dataloaders.dataloader_didemo_retrieval import DiDeMo_DataLoader
from dataloaders.dataloader_vatex_retrieval import VATEX_DataLoader
from dataloaders.dataloader_direction_mcq import DirectionMCQ_DataLoader


def dataloader_msrvtt_train(args, tokenizer):
    # print("args.features_path:")
    # print(args.features_path)
    # input()
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    else:
        train_sampler = None

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_hmdb_train(args, tokenizer, subset="test"):
    msrvtt_dataset = HMDB_DataLoader(
        json_path=args.train_csv,
        # json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        # unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    else:
        train_sampler = None

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_hmdb_test(args, tokenizer, subset="test"):
    msrvtt_testset = HMDB_DataLoader(
        json_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MSVD_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler

def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    dataloader_msrvtt = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msvd_testset)


def dataloader_lsmdc_train(args, tokenizer):
    lsmdc_dataset = LSMDC_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler

def dataloader_lsmdc_test(args, tokenizer, subset="test"):
    lsmdc_testset = LSMDC_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    dataloader_msrvtt = DataLoader(
        lsmdc_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(lsmdc_testset)


def dataloader_activity_train(args, tokenizer):
    activity_dataset = ActivityNet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activity_dataset), train_sampler

def dataloader_activity_test(args, tokenizer, subset="test"):
    activity_testset = ActivityNet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    dataloader_msrvtt = DataLoader(
        activity_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(activity_testset)


def dataloader_didemo_train(args, tokenizer):
    didemo_dataset = DiDeMo_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(didemo_dataset), train_sampler

def dataloader_didemo_test(args, tokenizer, subset="test"):
    didemo_testset = DiDeMo_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    dataloader_didemo = DataLoader(
        didemo_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_didemo, len(didemo_testset)

def dataloader_vatex_train(args, tokenizer):
    vatex_dataset = VATEX_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_dataset)
    dataloader = DataLoader(
        vatex_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(vatex_dataset), train_sampler

def dataloader_vatex_test(args, tokenizer, subset="test"):
    vatex_testset = VATEX_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        vatex_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(vatex_testset)

class NoRepeatBalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_classes=9, max_per_class=4, drop_last=True):
        self.labels = list(map(int, labels))
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.max_per_class = max_per_class
        self.drop_last = drop_last
        self.epoch = 0

        if batch_size < num_classes:
            raise ValueError(
                f"batch_size({batch_size}) must be >= num_classes({num_classes})."
            )

        if max_per_class * num_classes < batch_size:
            raise ValueError(
                f"max_per_class({max_per_class}) is too small for batch_size({batch_size}) "
                f"and num_classes({num_classes})."
            )

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        for c in range(num_classes):
            if len(self.class_to_indices[c]) == 0:
                raise ValueError(f"Class {c} has no samples.")

        if drop_last:
            self.num_batches = len(self.labels) // batch_size
        else:
            self.num_batches = (len(self.labels) + batch_size - 1) // batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.epoch)

        pools = {}
        for c in range(self.num_classes):
            indices = self.class_to_indices[c].copy()
            rng.shuffle(indices)
            pools[c] = indices

        while True:
            total_remaining = sum(len(pools[c]) for c in range(self.num_classes))

            if self.drop_last and total_remaining < self.batch_size:
                break

            if not self.drop_last and total_remaining == 0:
                break

            batch = []
            batch_class_count = {c: 0 for c in range(self.num_classes)}

            # 1) 가능한 한 모든 class에서 1개씩 먼저 뽑기
            classes = list(range(self.num_classes))
            rng.shuffle(classes)

            for c in classes:
                if len(batch) >= self.batch_size:
                    break

                if len(pools[c]) > 0:
                    batch.append(pools[c].pop())
                    batch_class_count[c] += 1

            # class가 너무 적게 남아서 모든 class를 못 넣는 후반부면 중단
            # 예: class 4만 남은 상태에서 batch 만드는 것 방지
            active_classes = sum(1 for c in range(self.num_classes) if batch_class_count[c] > 0)
            if active_classes < self.num_classes:
                break

            # 2) 남은 자리는 max_per_class 제한 안에서 채우기
            while len(batch) < self.batch_size:
                available = [
                    c for c in range(self.num_classes)
                    if len(pools[c]) > 0 and batch_class_count[c] < self.max_per_class
                ]

                # 중요:
                # 여기서 max_per_class 제한을 풀면 후반에 [0,0,0,0,16,0,0,0,0] 같은 batch가 나옴.
                # 그래서 제한을 풀지 않고 그냥 이 batch를 버림.
                if len(available) == 0:
                    break

                weights = [len(pools[c]) for c in available]
                c = rng.choices(available, weights=weights, k=1)[0]

                batch.append(pools[c].pop())
                batch_class_count[c] += 1

            if len(batch) == self.batch_size:
                rng.shuffle(batch)
                yield batch
            elif not self.drop_last and len(batch) > 0:
                rng.shuffle(batch)
                yield batch
                break
            else:
                break

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        random.seed(epoch)

def dataloader_direction_mcq_train(args, tokenizer):
    dataset = DirectionMCQ_DataLoader(
        json_path=args.train_csv,
        features_path=args.features_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    labels = [int(item["label"]) for item in dataset.data]

    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    print(f"[balanced_batch_sampler] class_counts = {dict(sorted(class_counts.items()))}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            raise NotImplementedError(
                "NoRepeatBalancedBatchSampler is currently for single-GPU / single-rank training only."
            )

    batch_sampler = NoRepeatBalancedBatchSampler(
        labels=labels,
        batch_size=args.batch_size,
        num_classes=9,
        max_per_class=4,
        drop_last=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    return dataloader, len(dataset), batch_sampler


def dataloader_direction_mcq_test(args, tokenizer, subset="test"):
    json_path = args.val_csv

    dataset = DirectionMCQ_DataLoader(
        json_path=json_path,
        features_path=args.features_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    return dataloader, len(dataset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {
    "train": dataloader_msrvtt_train,
    "val": dataloader_msrvtt_test,
    "test": None,
}
DATALOADER_DICT["hmdb"] = {
    "train": dataloader_hmdb_train,
    "val": dataloader_hmdb_test,
    "test": None,
}
DATALOADER_DICT["msvd"] = {
    "train": dataloader_msvd_train,
    "val": dataloader_msvd_test,
    "test": dataloader_msvd_test,
}
DATALOADER_DICT["lsmdc"] = {
    "train": dataloader_lsmdc_train,
    "val": dataloader_lsmdc_test,
    "test": dataloader_lsmdc_test,
}
DATALOADER_DICT["activity"] = {
    "train": dataloader_activity_train,
    "val": dataloader_activity_test,
    "test": None,
}
DATALOADER_DICT["didemo"] = {
    "train": dataloader_didemo_train,
    "val": dataloader_didemo_test,
    "test": dataloader_didemo_test,
}

DATALOADER_DICT["direction_mcq"] = {
    "train": dataloader_direction_mcq_train,
    "val": dataloader_direction_mcq_test,
    "test": dataloader_direction_mcq_test,
}
