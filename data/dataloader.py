import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import glob
import os


# class MRIDataset(Dataset):
#     def __init__(self, base_dir, transform=None):
#         # Define subdirectories for low-res and high-res
#         low_res_dir = os.path.join(base_dir, "Low-Res")
#         high_res_dir = os.path.join(base_dir, "High-Res")

#         # Collect files with specific keywords in their names
#         self.low_res_files = sorted(glob.glob(os.path.join(low_res_dir, "*.nii")))
#         self.high_res_files = sorted(glob.glob(os.path.join(high_res_dir, "*.nii")))

#         # Ensure we have pairs of files
#         if len(self.low_res_files) != len(self.high_res_files):
#             raise ValueError("Mismatch between number of low-res and high-res files.")

#         if len(self.low_res_files) == 0 or len(self.high_res_files) == 0:
#             raise ValueError("No files found. Please check the directory paths.")

#         self.transform = transform

#     def __len__(self):
#         return len(self.low_res_files)

#     def __getitem__(self, idx):
#         # Load low-res and high-res .nii.gz files
#         low_res_path = self.low_res_files[idx]
#         high_res_path = self.high_res_files[idx]

#         low_res_data = nib.load(low_res_path).get_fdata()
#         high_res_data = nib.load(high_res_path).get_fdata()

#         # Ensure dimensions are compatible: add a channel dimension [1, 30, 256, 256] and [1, 150, 256, 256]
#         low_res_data = torch.tensor(low_res_data, dtype=torch.float32).unsqueeze(0)
#         high_res_data = torch.tensor(high_res_data, dtype=torch.float32).unsqueeze(0)

#         # Apply any additional transformations if provided
#         if self.transform:
#             low_res_data = self.transform(low_res_data)
#             high_res_data = self.transform(high_res_data)

#         return low_res_data, high_res_data


class MRIDataset(Dataset):
    def __init__(self, txt_file, transform=None, limit=None):
        """
        Args:
            txt_file (string): Path to the text file with pairs of MRI file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): Maximum number of file pairs to load.
        """
        self.transform = transform
        self.limit = limit
        self.mri_pairs = self._read_txt_file(txt_file)

    def _read_txt_file(self, txt_file):
        """Reads a text file and returns a list of tuples (high_res_path, low_res_path)."""
        pairs = []
        with open(txt_file, "r") as file:
            lines = file.readlines()
            for line in lines[
                : self.limit
            ]:  # Apply the limit on the number of lines read
                high_res_path, low_res_path = line.strip().split(", ")
                pairs.append((high_res_path, low_res_path))
        return pairs

    def __len__(self):
        return len(self.mri_pairs)

    def __getitem__(self, idx):
        # Extract file paths for high-res and low-res images
        high_res_path, low_res_path = self.mri_pairs[idx]

        # Load MRI data from the NIfTI files
        high_res_data = nib.load(high_res_path).get_fdata()
        low_res_data = nib.load(low_res_path).get_fdata()

        # Convert numpy arrays to PyTorch tensors and add a channel dimension
        high_res_data = torch.tensor(high_res_data, dtype=torch.float32).unsqueeze(0)
        low_res_data = torch.tensor(low_res_data, dtype=torch.float32).unsqueeze(0)

        # Apply transformations if any
        if self.transform:
            high_res_data = self.transform(high_res_data)
            low_res_data = self.transform(low_res_data)

        return high_res_data, low_res_data
