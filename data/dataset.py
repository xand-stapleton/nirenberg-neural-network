import os
import tensorflow as tf
from tensorflow.data import Dataset

from data.prescribers import Prescriber
from data.samplers import Sampler


def build_dataset(
    patch_coords_key: str,
    label_key: str,
    normalization_key: str,
    sampler: Sampler,
    prescriber: Prescriber,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = 42,
) -> Dataset:
    """
    Build a TensorFlow dataset for training the Kazdan-Warner model.

    Samples points on the sphere using the provided sampler, evaluates the prescribed
    scalar curvature at those points, and packages the data into batches. The normalization
    factor (integral of the square of the prescribed curvature) is computed once and
    attached to each batch for loss normalization.

    Args:
        patch_coords_key: key for patch coordinates in the batch dictionary
        label_key: key for prescribed scalar curvature labels
        normalization_key: key for the normalization factor
        sampler: sampler for generating training points
        prescriber: prescriber function for computing scalar curvature labels
        batch_size: number of samples per batch
        shuffle: whether to shuffle the dataset
        seed: random seed for shuffling

    Returns:
        Dataset: TensorFlow dataset yielding batches of training data

    """
    # Sample patch(es): chart coordinates and embedded coordinates
    vw, xyz = sampler()
    # Compute regression labels
    prescribed_R = prescriber(xyz)
    # Package into a data dict, to be divided into batches
    data_dict = {
        patch_coords_key: vw,
        label_key: prescribed_R,
    }

    # Construct tf.data.Dataset
    ds = Dataset.from_tensor_slices(data_dict)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=sampler.num_samples,
            seed=seed,
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(batch_size, drop_remainder=False)
    # Compute loss normalization factor once and attach to each emitted batch
    # If integral is too small (e.g., for zero prescriber), set to 1.0 (no normalization)
    norm_value = prescriber.integrate()
    if norm_value < 1e-10:
        norm_value = 1.0
    norm_scalar = tf.constant([[norm_value]], dtype=xyz.dtype)
    ds = ds.map(lambda batch: {**batch, normalization_key: norm_scalar})
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def list_saved_models(import_from_seed_models: bool = False):
    """
    List available Keras model files.
    - If import_from_seed_models=True: list .keras files in 'seed_models' (flat structure).
    - If import_from_seed_models=False: list .keras files in each subfolder of 'checkpoints/'.
    Returns:
        - For seed models: list of model filenames (no extension)
        - For checkpoints: dict {run_folder: [model_file_paths]}
    """
    base_dir = os.getcwd()
    if import_from_seed_models:
        root_runs_path = os.path.join(base_dir, "seed_models")
        if not os.path.exists(root_runs_path):
            print("No seed models found...")
            return []
        saved_models = [
            f for f in os.listdir(root_runs_path)
            if os.path.isfile(os.path.join(root_runs_path, f)) and f.endswith(".keras")
        ]
        saved_models = [os.path.splitext(f)[0] for f in saved_models]
        print(f"Available models in seed_models:")
        for idx, name in enumerate(saved_models):
            print(f"{idx}: {name}")
        return saved_models
    else:
        checkpoints_dir = os.path.join(base_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            print("No completed checkpoints...")
            return []
        run_folders = [
            f for f in os.listdir(checkpoints_dir)
            if os.path.isdir(os.path.join(checkpoints_dir, f))
        ]
        run_folders.sort()
        print("Available checkpoint folders:")
        for idx, run_folder in enumerate(run_folders):
            print(f"  [{idx}] {run_folder}")
        return run_folders
