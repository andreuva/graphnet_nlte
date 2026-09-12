# refactored_synchronized_cleaner.py
import numpy as np
import pickle
from glob import glob
import os

def load_all_datasets(filepaths: list) -> list:
    """Loads all pickle files into a list of datasets."""
    all_datasets = []
    for filepath in filepaths:
        try:
            with open(filepath, 'rb') as f:
                dataset = pickle.load(f)
                if not isinstance(dataset, list):
                    print(f"Warning: Content of {os.path.basename(filepath)} is not a list. Skipping.")
                    continue
                all_datasets.append(dataset)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error reading {os.path.basename(filepath)}. It may be corrupt. Skipping. Details: {e}")
    return all_datasets

def find_global_invalid_indices(all_datasets: list) -> set:
    """
    Finds the union of all indices corresponding to invalid data (None or NaN)
    across all datasets.

    Args:
        all_datasets: A list where each element is a dataset (a list of items).

    Returns:
        A set of integer indices that should be removed from all datasets.
    """
    invalid_indices = set()
    
    # Check that all datasets have the same length before starting
    if len(set(len(d) for d in all_datasets)) > 1:
        print("Warning: Datasets have inconsistent lengths. Synchronization may be incorrect.")
        # You might want to raise an error here depending on how strict you need to be
        # raise ValueError("Inconsistent dataset lengths found.")

    for dataset in all_datasets:
        for i, item in enumerate(dataset):
            if item is None:
                invalid_indices.add(i)
            else:
                try:
                    if not np.all(np.isfinite(item)):
                        invalid_indices.add(i)
                except Exception:
                    pass
    return invalid_indices

def clean_and_save_datasets(filepaths: list, all_datasets: list, invalid_indices: set):
    """
    Removes items at the specified indices from all datasets and saves them.
    
    Args:
        filepaths: The original file paths.
        all_datasets: The list of datasets to clean.
        invalid_indices: A set of indices to remove.
    """
    if not invalid_indices:
        print("✅ No invalid entries found across all files. All datasets are clean and synchronized.")
        return

    print(f"Found {len(invalid_indices)} unique indices to remove across all files.")

    for i, (filepath, original_dataset) in enumerate(zip(filepaths, all_datasets)):
        # Build the new dataset using a list comprehension for safety and efficiency
        cleaned_dataset = [
            item for idx, item in enumerate(original_dataset)
            if idx not in invalid_indices
        ]
        
        print(f"  Cleaning {os.path.basename(filepath)}... Original size: {len(original_dataset)}, New size: {len(cleaned_dataset)}")
        
        # Overwrite the original file with the cleaned, synchronized data
        with open(filepath, 'wb') as f_out:
            pickle.dump(cleaned_dataset, f_out)
    
    print("\n✅ All files have been cleaned and synchronized.")

def main():
    """
    Main function to find, synchronize, clean, and save all dataset files.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Synchronously clean datasets by removing indices with NaNs or Nones.")
    parser.add_argument('--dir', default='../data_1d_adur/', help='Directory containing pickle files')
    args = parser.parse_args()

    datadir = args.dir
    if not datadir.endswith('/'):
        datadir += '/'

    # Find all prefixes by looking for files ending with _T.pkl
    t_files = glob(os.path.join(datadir, '*_T.pkl'))
    prefixes = sorted(list(set(os.path.basename(f).split('_')[0] for f in t_files)))

    if not prefixes:
        print(f"No datasets found in {datadir}")
        return

    print(f"Starting synchronized dataset cleaning process in: {datadir}")
    print(f"Found prefixes to clean: {prefixes}")

    for prefix in prefixes:
        print(f"\n--- Processing prefix: {prefix} ---")
        file_pattern = os.path.join(datadir, f'{prefix}_*.pkl')
        filepaths = glob(file_pattern)
        
        # Filter filepaths to only include those ending in known patterns to avoid cleaning unrelated files
        valid_patterns = ['_T.pkl', '_tau.pkl', '_ne.pkl', '_vturb.pkl', '_vlos.pkl', '_z.pkl', '_logdeparture.pkl', '_n_Nat.pkl', '_Iwave.pkl']
        filepaths = sorted([f for f in filepaths if any(f.endswith(p) for p in valid_patterns)])

        if not filepaths:
            print(f"No files found for prefix '{prefix}'")
            continue

        print(f"Found {len(filepaths)} files to process for prefix '{prefix}':")
        for fp in filepaths:
            print(f"  - {os.path.basename(fp)}")

        # PASS 1: Load data and find all bad indices
        datasets = load_all_datasets(filepaths)
        if not datasets:
            print("No valid datasets were loaded. Skipping.")
            continue
            
        global_invalid_indices = find_global_invalid_indices(datasets)

        # PASS 2: Clean all datasets using the global index list and save them
        clean_and_save_datasets(filepaths, datasets, global_invalid_indices)

if __name__ == "__main__":
    main()