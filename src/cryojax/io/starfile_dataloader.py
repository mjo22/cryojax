import os
import starfile
import mrcfile
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils import data
import jax.numpy as jnp
from jax.tree_util import tree_map


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


class RelionDataLoader(Dataset):
    def __init__(self, data_path: str, name_star_file: str):
        self.data_path = data_path
        self.name_star_file = name_star_file

        df = starfile.read(os.path.join(self.data_path, self.name_star_file))

        if isinstance(df, dict):
            if "particles" in df:
                self.df = df["particles"]
                self.num_projs = len(df["particles"])
                self.optic_params = df["optics"]

        else:
            self.df = df
            self.num_projs = len(self.df)
            self.optic_params = None
            raise Warning("No optics parameters found in starfile")

        # self.vol_sidelen = self.df["optics"]["rlnImageSize"][0]

    def get_df_optics_params(self):
        if self.optic_params is None:
            raise Exception("No optics parameters found in star file")
        else:
            return (
                self.df["optics"]["rlnImageSize"][0],
                self.df["optics"]["rlnVoltage"][0],
                self.df["optics"]["rlnImagePixelSize"][0],
                self.df["optics"]["rlnSphericalAberration"][0],
                self.df["optics"]["rlnAmplitudeContrast"][0],
            )

    def __len__(self):
        return self.num_projs

    def __getitem__(self, idx):
        particle = self.df.iloc[idx]
        try:
            # Load particle image from mrcs file
            imgnamedf = particle["rlnImageName"].split("@")
            mrc_path = os.path.join(self.data_path, imgnamedf[1])
            pidx = int(imgnamedf[0]) - 1
            with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
                proj = mrc.data[pidx]
        except:
            raise Exception("Error loading image from mrcs file")

        # Generate CTF from relion paramaters
        defocus_u = np.array(particle["rlnDefocusU"])
        defocus_v = np.array(particle["rlnDefocusV"])
        angleAstigmatism = np.radians(np.array(particle["rlnDefocusAngle"]))

        # Read relion orientations and shifts
        pose = np.zeros(5)
        pose[0] = np.array(particle["rlnOriginXAngst"])
        pose[1] = np.array(particle["rlnOriginYAngst"])
        pose[2:] = np.radians(
            np.stack(
                [
                    particle["rlnAnglePsi"],
                    particle["rlnAngleTilt"],
                    particle["rlnAngleRot"],
                ]
            )
        )

        img_params = {
            "proj": proj,
            "pose": pose,
            "defocus_u": defocus_u,
            "defocus_v": defocus_v,
            "angleAstigmatism": angleAstigmatism,
            "idx": idx,
        }

        return img_params


def load_starfile(data_path: str, name_star_file: str, batch_size: int):
    """
    Load relion starfile into torch dataloader adapted to numpy arrays

    Parameters
    ----------
    data_path : str
        Path to starfile and mrcs files
    name_star_file : str
        Name of starfile
    batch_size : int
        Batch size for dataloader

    Returns
    -------
    dataloader : torch dataloader
        Dataloader with numpy arrays
    """
    image_stack = RelionDataLoader(
        data_path=data_path,
        name_star_file=name_star_file,
    )

    dataloader = NumpyLoader(
        image_stack,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )

    return dataloader
