import copy
import csv
from datetime import datetime
from typing import Sized

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class SequentialBatchSampler(Sampler[int]):
    data_source: Sized

    def __init__(
        self,
        data: Sized,
        meta_data_path: str,
        seq_length=1,
        shuffle: bool = False,
        step: int = 10,
        stage: str = "train",
        id_stepping: str = "multi",
    ) -> None:
        super().__init__(data)
        self.data = data
        self.meta_data_path = meta_data_path
        self.step = step
        self.meta_data: np.ndarray = self.load_csv(self.meta_data_path, "	").to_numpy()[
            :: self.step, :
        ]
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.cleaned_meta_data = copy.deepcopy(self.meta_data)
        self.stage = stage
        self.id_stepping = id_stepping

        # switch between sliding window modes
        if self.id_stepping == "single":
            # images can only appear in a single sequence
            stepping = self.seq_length
        elif self.id_stepping == "multi":
            # images can appear in multiple sequences
            stepping = 1
        else:
            raise ValueError("id_stepping must be either 'single' or 'multi'")

        # generate image id sequences
        self.seq_chunks = np.array(
            [
                range(i, i + self.seq_length)
                for i in range(0, (len(self.data) - (self.seq_length - 1)), stepping)
            ]
        )
        border_frame_ids = self.get_video_borders()

        bad_seq_ids = []
        bad_sequences = []
        for seq_i, seq in enumerate(self.seq_chunks):
            # check if border is in the sequences
            for border in border_frame_ids:
                if border in seq:
                    # border (= last frame of a video) is in seq so check if it is not the last item of the seq
                    if seq[-1] != border:
                        # remove this sequence as it overlaps videos
                        # save sequences to remove them from sequence list and metadata
                        bad_seq_ids.append(seq_i)
                        rem_seq = self.seq_chunks[seq_i]
                        bad_sequences.append(rem_seq)
        if self.id_stepping == "single":
            self.seq_chunks = np.delete(self.seq_chunks, bad_seq_ids, axis=0)
            self.cleaned_meta_data = np.delete(
                self.cleaned_meta_data, np.array(bad_sequences).flatten(), 0
            )
        else:
            raise NotImplementedError("id_stepping='multi' not implemented yet")

        # write updated metadata data during test
        if self.stage == "test":
            self.write_csv(
                self.cleaned_meta_data, np.array([[seq_id] for seq_id in bad_seq_ids])
            )

    @staticmethod
    def load_csv(embed_path, delimiter, header="infer"):
        data = pd.read_csv(
            embed_path, delimiter=delimiter, header=header, skip_blank_lines=False
        )
        return data

    def write_csv(self, data, ids):
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        with open(
            f"latents/{dt_string}_{self.id_stepping}_cleaned_meta_data.csv", "w"
        ) as csvfile:
            writer = csv.writer(csvfile, delimiter="	")
            # writer.writerow("ID	Mutation	Dir	File")  # header
            writer.writerows(data)
        with open(
            f"latents/{dt_string}_{self.id_stepping}_removed_seq_ids.csv", "w"
        ) as csvfile:
            writer = csv.writer(csvfile, delimiter="	")
            # writer.writerow("ID	Mutation	Dir	File")  # header
            writer.writerows(ids)

    def get_video_borders(self) -> list[int]:
        seen_files = []
        border_frame_ids = []  # id of last frame of a video
        for row_id, entry in reversed(list(enumerate(self.meta_data))):
            # check if file is already in list. Decision is made on video basis (not mutant basis)
            if entry[2] not in seen_files:
                seen_files.append(entry[2])  # add video file name
                border_frame_ids.append(row_id)  # add iterator frame id

        return border_frame_ids

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.seq_chunks)
        return iter(self.seq_chunks)

    def __len__(self) -> int:
        return len(self.seq_chunks)
