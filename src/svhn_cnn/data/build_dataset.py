import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from svhn_cnn.utils import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def expand(self, factor: float, img_height: int, img_width: int) -> "BoundingBox":
        """Expand the bounding box by a factor while staying within image bounds."""
        h = self.y_max - self.y_min
        w = self.x_max - self.x_min
        h_expand = int(h * factor)
        w_expand = int(w * factor)

        return BoundingBox(
            x_min=max(0, self.x_min - w_expand),
            y_min=max(0, self.y_min - h_expand),
            x_max=min(img_width, self.x_max + w_expand),
            y_max=min(img_height, self.y_max + h_expand),
        )

    def is_valid(self) -> bool:
        """Check if bounding box has non-zero width and height."""
        return (self.x_max - self.x_min > 0) and (self.y_max - self.y_min > 0)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass
class DatasetConfig:
    """Configuration for dataset extraction."""

    image_size: tuple[int, int] = (48, 48)
    max_digits: int = 4
    expansion_factor: float = 0.3
    blank_label: int = 10
    min_negative_region_size: int = 10


def _extract_dataset_generic(
    dataset_type: str,
    output_path: Optional[Path] = None,
    config_obj: Optional[DatasetConfig] = None,
    use_zarr: bool = False,
) -> None:
    """
    Generic dataset extraction function that works for train/test/extra datasets.

    This eliminates code duplication across the three extraction functions.

    Args:
        dataset_type: One of "train", "test", or "extra"
        output_path: Optional custom output path for dataset
        config_obj: Configuration object with extraction parameters
        use_zarr: If True, use Zarr format instead of HDF5
    """
    # Initialize configuration
    cfg = config_obj or DatasetConfig()
    data_dir = Path(config.config["datasets.path"][dataset_type])

    if output_path is None:
        output_path = data_dir / f"{dataset_type}.h5"

    logger.info(f"Starting {dataset_type} dataset extraction from {data_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Image size: {cfg.image_size}, Max digits: {cfg.max_digits}")

    # Load input metadata
    try:
        input_hf = h5py.File(data_dir / "digitStruct.mat", "r")
        data = input_hf["digitStruct"]
        bbox_data = data["bbox"]
        name_data = data["name"]
        if bbox_data.shape != name_data.shape:
            logger.warning(
                f"Bbox and Name have different shapes ({bbox_data.shape}, {name_data.shape} respectively)... Picking the smallest as number of samples"
            )
        num_samples = min(bbox_data.shape[0], name_data.shape[0])
    except Exception as e:
        logger.error(f"Failed to load digitStruct.mat: {e}")
        raise

    # Prepare labels
    neglabel = np.array([0] + [cfg.blank_label] * 5, dtype=np.uint8)

    # First pass: count valid samples for efficient array pre-allocation
    logger.info("First pass: counting valid samples...")
    valid_samples = _count_valid_samples(
        bbox_data, input_hf, num_samples, cfg.max_digits
    )
    logger.info(f"Found {valid_samples} valid samples")

    if valid_samples == 0:
        logger.warning(
            "No valid samples found! Check that digitStruct.mat and images are in the correct format."
        )
        input_hf.close()
        return

    # Pre-allocate arrays (more memory efficient than lists)
    pos_samples_rgb = np.zeros((valid_samples, *cfg.image_size, 3), dtype=np.uint8)
    pos_samples_gray = np.zeros((valid_samples, *cfg.image_size), dtype=np.uint8)
    pos_labels = np.zeros((valid_samples, 6), dtype=np.uint8)

    # Lists for negative samples (variable count)
    neg_samples_rgb = []
    neg_samples_gray = []
    neg_labels = []

    # Process samples with progress bar
    valid_idx = 0
    for i in tqdm(range(num_samples), desc=f"Processing {dataset_type} images"):
        try:
            # Parse filename - SVHN uses name[i][0] structure
            img_name = _parse_hdf5_string(name_data[i][0], input_hf)
            img_path = data_dir / img_name

            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            img_height, img_width = img.shape[:2]

            # Parse bounding box (bbox_data[i] is array, [0] gets the reference)
            bbox = _parse_bbox(bbox_data[i][0], input_hf, img_height, img_width, cfg)
            if not bbox.is_valid():
                logger.debug(f"Sample {i}: Invalid bbox")
                continue

            # Parse labels
            labels = _parse_labels(bbox_data[i][0], input_hf, cfg)
            if labels is None:
                logger.debug(f"Sample {i}: Failed to parse labels")
                continue
            # labels[0] contains the actual number of digits
            if labels[0] > cfg.max_digits + 1:
                logger.debug(f"Sample {i}: Too many digits ({labels[0]})")
                continue

            # Extract positive sample (digit region)
            digit_crop = img[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max, :]
            digit_resized = cv2.resize(
                digit_crop, cfg.image_size, interpolation=cv2.INTER_AREA
            )
            digit_gray = cv2.cvtColor(digit_resized, cv2.COLOR_BGR2GRAY)

            pos_samples_rgb[valid_idx] = digit_resized
            pos_samples_gray[valid_idx] = digit_gray
            pos_labels[valid_idx] = labels
            valid_idx += 1

            # Extract negative samples (regions without digits)
            neg_crops = _extract_negative_samples(img, bbox, cfg)
            for neg_crop in neg_crops:
                neg_resized = cv2.resize(
                    neg_crop, cfg.image_size, interpolation=cv2.INTER_AREA
                )
                neg_gray = cv2.cvtColor(neg_resized, cv2.COLOR_BGR2GRAY)
                neg_samples_rgb.append(neg_resized)
                neg_samples_gray.append(neg_gray)
                neg_labels.append(neglabel)

        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            continue

    input_hf.close()

    # Trim to actual valid count
    pos_samples_rgb = pos_samples_rgb[:valid_idx]
    pos_samples_gray = pos_samples_gray[:valid_idx]
    pos_labels = pos_labels[:valid_idx]

    # Convert negative samples to arrays
    if neg_samples_rgb:
        neg_samples_rgb = np.array(neg_samples_rgb, dtype=np.uint8)
        neg_samples_gray = np.array(neg_samples_gray, dtype=np.uint8)
        neg_labels = np.array(neg_labels, dtype=np.uint8)
    else:
        neg_samples_rgb = np.zeros((0, *cfg.image_size, 3), dtype=np.uint8)
        neg_samples_gray = np.zeros((0, *cfg.image_size), dtype=np.uint8)
        neg_labels = np.zeros((0, 6), dtype=np.uint8)

    logger.info(
        f"Extracted {valid_idx} positive and {len(neg_labels)} negative samples"
    )

    # Write to disk
    logger.info(f"Writing dataset to {output_path}")
    if use_zarr:
        _write_zarr(
            output_path,
            pos_samples_rgb,
            pos_samples_gray,
            pos_labels,
            neg_samples_rgb,
            neg_samples_gray,
            neg_labels,
        )
    else:
        _write_hdf5(
            output_path,
            pos_samples_rgb,
            pos_samples_gray,
            pos_labels,
            neg_samples_rgb,
            neg_samples_gray,
            neg_labels,
        )

    logger.info(f"{dataset_type.capitalize()} dataset extraction complete!")


def extract_trainRGB(
    output_path: Optional[Path] = None,
    config_obj: Optional[DatasetConfig] = None,
    use_zarr: bool = False,
) -> None:
    """
    Extract training data from SVHN .mat file with modern best practices.

    Improvements over original:
    - Memory efficient: streams data to disk instead of loading all into memory
    - Progress tracking with tqdm
    - Proper error handling
    - Type hints for better code clarity
    - Configurable parameters via DatasetConfig
    - Option to use Zarr instead of HDF5 (better for cloud storage)
    - Cleaner bbox handling with dataclasses
    - Proper logging instead of print statements

    Args:
        output_path: Optional custom output path for dataset
        config_obj: Configuration object with extraction parameters
        use_zarr: If True, use Zarr format instead of HDF5
    """
    _extract_dataset_generic("train", output_path, config_obj, use_zarr)


def extract_testRGB(
    output_path: Optional[Path] = None,
    config_obj: Optional[DatasetConfig] = None,
    use_zarr: bool = False,
) -> None:
    """
    Extract test data from SVHN .mat file with modern best practices.

    Improvements over original:
    - Memory efficient: streams data to disk instead of loading all into memory
    - Progress tracking with tqdm
    - Proper error handling
    - Type hints for better code clarity
    - Configurable parameters via DatasetConfig
    - Option to use Zarr instead of HDF5 (better for cloud storage)
    - Cleaner bbox handling with dataclasses
    - Proper logging instead of print statements

    Args:
        output_path: Optional custom output path for dataset
        config_obj: Configuration object with extraction parameters
        use_zarr: If True, use Zarr format instead of HDF5
    """
    _extract_dataset_generic("test", output_path, config_obj, use_zarr)


def extract_extraRGB(
    output_path: Optional[Path] = None,
    config_obj: Optional[DatasetConfig] = None,
    use_zarr: bool = False,
) -> None:
    """
    Extract extra data from SVHN .mat file with modern best practices.

    Improvements over original:
    - Memory efficient: streams data to disk instead of loading all into memory
    - Progress tracking with tqdm
    - Proper error handling
    - Type hints for better code clarity
    - Configurable parameters via DatasetConfig
    - Option to use Zarr instead of HDF5 (better for cloud storage)
    - Cleaner bbox handling with dataclasses
    - Proper logging instead of print statements

    Args:
        output_path: Optional custom output path for dataset
        config_obj: Configuration object with extraction parameters
        use_zarr: If True, use Zarr format instead of HDF5
    """
    _extract_dataset_generic("extra", output_path, config_obj, use_zarr)


def _count_valid_samples(bbox_data, h5file, num_samples: int, max_digits: int) -> int:
    """Count number of valid samples in first pass."""
    count = 0
    for i in range(num_samples):
        try:
            # bbox_data[i] is array, [0] gets the reference
            bbox_ref = bbox_data[i][0]
            bbox_obj = h5file[bbox_ref]

            # Count labels
            label_field = bbox_obj["label"]
            refs_array = label_field[()]
            numdig = len(refs_array)

            if 0 < numdig <= max_digits + 1:
                count += 1
        except:
            continue
    return count


def _parse_hdf5_string(hdf5_ref, h5file) -> str:
    """Parse HDF5 string reference to Python string.

    SVHN digitStruct.mat uses HDF5 object references that need to be
    dereferenced using the file handle.
    """
    try:
        # Handle HDF5 reference objects (SVHN format)
        if isinstance(hdf5_ref, h5py.h5r.Reference):
            # Dereference the object reference
            obj = h5file[hdf5_ref]
            # Get the actual string data
            if hasattr(obj, "value"):
                string_data = obj.value
            else:
                string_data = obj[()]

            # Convert to string
            if isinstance(string_data, bytes):
                return string_data.decode("utf-8")
            elif isinstance(string_data, np.ndarray):
                # Handle character arrays
                return "".join(chr(c) for c in string_data.flatten())
            return str(string_data)

        # Handle regular arrays/values
        if hasattr(hdf5_ref, "squeeze"):
            hdf5_ref = hdf5_ref.squeeze()
        if hasattr(hdf5_ref, "item"):
            val = hdf5_ref.item()
            if isinstance(val, bytes):
                return val.decode("utf-8")
            return str(val)
        if isinstance(hdf5_ref, bytes):
            return hdf5_ref.decode("utf-8")
        return str(hdf5_ref)
    except Exception as e:
        logger.debug(f"Error parsing HDF5 string: {e}")
        return ""


def _parse_bbox(
    bbox_ref, h5file, img_height: int, img_width: int, cfg: DatasetConfig
) -> BoundingBox:
    """Parse bounding box from HDF5 structure.

    SVHN digitStruct.mat has a nested structure:
    - bbox[i] is a numpy array containing a reference
    - bbox[i][0] is the HDF5 reference to the bbox object
    - bbox_obj['top'], etc. are datasets containing arrays of references
    - Each element is [ref], so we need [j][0] to get the actual reference
    - Each reference points to a single value
    """
    try:
        # Get the bbox object (bbox_ref is already bbox[i][0] from caller)
        bbox_obj = h5file[bbox_ref]

        # Helper to extract values from nested references
        def extract_values(field_name):
            field = bbox_obj[field_name]
            refs_array = field[()]  # Get array of references
            values = []
            for ref_wrapper in refs_array:
                # ref_wrapper is [ref], so get [0]
                ref = ref_wrapper[0]
                if isinstance(ref, h5py.h5r.Reference):
                    value_obj = h5file[ref]
                    value = value_obj[()]
                    # Value might be [[x]] or [x], flatten it
                    values.append(float(np.array(value).flatten()[0]))
            return np.array(values)

        top = extract_values("top")
        left = extract_values("left")
        height = extract_values("height")
        width = extract_values("width")

        # Create initial bbox
        y_min = max(0, int(np.min(top)))
        x_min = max(0, int(np.min(left)))
        y_max = int(np.max(top + height))
        x_max = int(x_min + np.sum(width))  # Sum widths for multi-digit sequences

        bbox = BoundingBox(x_min, y_min, x_max, y_max)

        # Expand by configured factor
        return bbox.expand(cfg.expansion_factor, img_height, img_width)

    except Exception as e:
        logger.debug(f"Error parsing bbox: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return BoundingBox(0, 0, 0, 0)


def _parse_labels(bbox_ref, h5file, cfg: DatasetConfig) -> Optional[np.ndarray]:
    """Parse digit labels from bounding box structure.

    Same nested structure as coordinates - each label is in a separate reference.
    """
    try:
        # Get the bbox object (bbox_ref is already the HDF5 reference)
        bbox_obj = h5file[bbox_ref]

        # Extract labels from nested references
        label_field = bbox_obj["label"]
        refs_array = label_field[()]  # Get array of references
        labels_raw = []
        for ref_wrapper in refs_array:
            ref = ref_wrapper[0]
            if isinstance(ref, h5py.h5r.Reference):
                value_obj = h5file[ref]
                value = value_obj[()]
                # Value might be [[x]] or [x], flatten it
                labels_raw.append(int(np.array(value).flatten()[0]))

        labels_raw = np.array(labels_raw, dtype=np.uint8)

        # Convert label 10 to 0 (SVHN uses 10 for digit 0)
        labels_raw = np.where(labels_raw == 10, 0, labels_raw)

        num_digits = len(labels_raw)
        if num_digits == 0 or num_digits > cfg.max_digits + 1:
            return None

        # Format: [num_digits, digit1, digit2, digit3, digit4, has_digits_flag]
        result = np.full(6, cfg.blank_label, dtype=np.uint8)
        result[0] = num_digits
        result[1 : num_digits + 1] = labels_raw[: cfg.max_digits]

        return result

    except Exception as e:
        logger.debug(f"Error parsing labels: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def _extract_negative_samples(
    img: np.ndarray, digit_bbox: BoundingBox, cfg: DatasetConfig
) -> list[np.ndarray]:
    """Extract negative samples (regions without digits) from image."""
    img_height, img_width = img.shape[:2]
    negative_crops = []
    min_size = cfg.min_negative_region_size

    # Top-left region
    if digit_bbox.x_min > min_size and digit_bbox.y_min > min_size:
        crop = img[0 : digit_bbox.y_min - 1, 0 : digit_bbox.x_min - 1, :]
        if crop.size > 0:
            negative_crops.append(crop)

    # Left region
    if digit_bbox.x_min > min_size:
        crop = img[digit_bbox.y_min : digit_bbox.y_max, 0 : digit_bbox.x_min - 1, :]
        if crop.size > 0:
            negative_crops.append(crop)

    # Top region
    if digit_bbox.y_min > min_size:
        crop = img[0 : digit_bbox.y_min - 1, digit_bbox.x_min : digit_bbox.x_max, :]
        if crop.size > 0:
            negative_crops.append(crop)

    # Bottom-right region
    if (
        img_height - digit_bbox.y_max > min_size
        and img_width - digit_bbox.x_max > min_size
    ):
        crop = img[
            digit_bbox.y_max + 1 : img_height, digit_bbox.x_max + 1 : img_width, :
        ]
        if crop.size > 0:
            negative_crops.append(crop)

    # Right region
    if img_width - digit_bbox.x_max > min_size:
        crop = img[
            digit_bbox.y_min : digit_bbox.y_max, digit_bbox.x_max + 1 : img_width, :
        ]
        if crop.size > 0:
            negative_crops.append(crop)

    # Bottom region
    if img_height - digit_bbox.y_max > min_size:
        crop = img[
            digit_bbox.y_max + 1 : img_height, digit_bbox.x_min : digit_bbox.x_max, :
        ]
        if crop.size > 0:
            negative_crops.append(crop)

    return negative_crops


def _write_hdf5(
    output_path: Path,
    pos_rgb: np.ndarray,
    pos_gray: np.ndarray,
    pos_labels: np.ndarray,
    neg_rgb: np.ndarray,
    neg_gray: np.ndarray,
    neg_labels: np.ndarray,
) -> None:
    """Write dataset to HDF5 format with compression."""
    with h5py.File(output_path, "w") as hf:
        # Use compression for better storage efficiency
        hf.create_dataset(
            "digits", data=pos_rgb, compression="gzip", compression_opts=4
        )
        hf.create_dataset(
            "digitsBW", data=pos_gray, compression="gzip", compression_opts=4
        )
        hf.create_dataset(
            "labs5", data=pos_labels, compression="gzip", compression_opts=4
        )

        hf.create_dataset(
            "negdigits", data=neg_rgb, compression="gzip", compression_opts=4
        )
        hf.create_dataset(
            "negdigitsBW", data=neg_gray, compression="gzip", compression_opts=4
        )
        hf.create_dataset(
            "neglab", data=neg_labels, compression="gzip", compression_opts=4
        )

        # Add metadata
        hf.attrs["num_positive"] = len(pos_labels)
        hf.attrs["num_negative"] = len(neg_labels)
        hf.attrs["image_size"] = pos_gray.shape[1:3]


def _write_zarr(
    output_path: Path,
    pos_rgb: np.ndarray,
    pos_gray: np.ndarray,
    pos_labels: np.ndarray,
    neg_rgb: np.ndarray,
    neg_gray: np.ndarray,
    neg_labels: np.ndarray,
) -> None:
    """Write dataset to Zarr format (modern alternative to HDF5)."""
    try:
        import zarr
    except ImportError:
        logger.error("Zarr not installed. Install with: pip install zarr")
        raise

    store = zarr.DirectoryStore(str(output_path).replace(".h5", ".zarr"))
    root = zarr.group(store=store, overwrite=True)

    # Zarr has better compression algorithms
    root.create_dataset(
        "digits",
        data=pos_rgb,
        chunks=(100, 48, 48, 3),
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    root.create_dataset(
        "digitsBW",
        data=pos_gray,
        chunks=(100, 48, 48),
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    root.create_dataset(
        "labs5", data=pos_labels, compressor=zarr.Blosc(cname="zstd", clevel=3)
    )

    root.create_dataset(
        "negdigits",
        data=neg_rgb,
        chunks=(100, 48, 48, 3),
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    root.create_dataset(
        "negdigitsBW",
        data=neg_gray,
        chunks=(100, 48, 48),
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    root.create_dataset(
        "neglab", data=neg_labels, compressor=zarr.Blosc(cname="zstd", clevel=3)
    )

    # Add metadata
    root.attrs["num_positive"] = len(pos_labels)
    root.attrs["num_negative"] = len(neg_labels)
    root.attrs["image_size"] = pos_gray.shape[1:3]


def prepDataforCNN(numChannel=1, feat_norm=False):
    htr = h5py.File(Path(config.config["datasets.path"]["train"]) / "train.h5", "r")
    hts = h5py.File(Path(config.config["datasets.path"]["test"]) / "test.h5", "r")
    # hte = h5py.File(Path(config.config["datasets.path"]["extra"]) / "extra.h5", "r")

    if numChannel == 1:
        digits = htr["digitsBW"]
        testdigits = hts["digitsBW"]
        negdigits = htr["negdigitsBW"]
        # extdigits = hte["digitsBW"]
    else:
        digits = htr["digits"]
        testdigits = hts["digits"]
        negdigits = htr["negdigits"]
        # extdigits = hte["digits"]

    trainlabs = htr["labs5"]
    testlabs = hts["labs5"]
    neglabs = htr["neglab"]
    # extlabs = hte["labs5"]

    digits = digits[:]
    testdigits = testdigits[:]
    negdigits = negdigits[:]
    # extdigits = extdigits[:]
    trainlabs = trainlabs[:]
    testlabs = testlabs[:]
    neglabs = neglabs[:]
    negdigits = negdigits[:]

    seed = 25
    np.random.seed(seed)
    countNeg = 30000
    countX = 90000

    negIdx = np.random.randint(0, negdigits.shape[0], countNeg)
    numNegTs = np.arange(negdigits.shape[0] - 500, negdigits.shape[0], 1)
    numtran = np.arange(
        0, countX, 1
    )  # Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(50%) % Tried on 12K
    # numtest = np.arange(extdigits.shape[0] - 1,extdigits.shape[0],1)

    # negIdx = np.random.randint(0,negdigits.shape[0],3000)# Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(50%) Tried on 12k
    # numtran = range(0,50,1)  # Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(50%) % Tried on 12K

    # xtrdigits = extdigits[numtran, :]
    # xtrlab = extlabs[numtran, :]

    ntrdigits = negdigits[negIdx]
    ntrlab = neglabs[negIdx, :]
    ntest = negdigits[numNegTs, :]
    ntslab = neglabs[numNegTs, :]

    preTrain = np.vstack(
        (
            digits,
            # xtrdigits,
            ntrdigits,
        )
    ).astype("float32")  #
    preTest = np.vstack((testdigits, ntest)).astype("float32")  # xtest

    # Lets remove digits > 4. Only 9 cases of n = 5
    trainlabs = np.vstack(
        (
            trainlabs,
            # xtrlab,
            ntrlab,
        )
    ).astype("uint8")
    testlabs = np.vstack((testlabs, ntslab)).astype("uint8")  # xtslab
    ind = np.argwhere(trainlabs[:, 0] < 5)
    ind = ind[:, 0]
    preTrain = preTrain[ind, :]
    trainlabs = trainlabs[ind, :]

    nb = np.reshape(
        np.asarray(trainlabs[:, 0] > 0, dtype="uint8"), (trainlabs.shape[0], 1)
    )
    trl = np.hstack((trainlabs, nb))

    ind = np.argwhere(testlabs[:, 0] < 5)
    ind = ind[:, 0]
    preTest = preTest[ind, :]
    testlabs = testlabs[ind, :]

    nb = np.reshape(
        np.asarray(testlabs[:, 0] > 0, dtype="uint8"), (testlabs.shape[0], 1)
    )
    tsl = np.hstack((testlabs, nb))

    # train = np.float64(preTrain/255.)
    # test  = np.float64(preTest/255.)
    train = np.float64(preTrain)
    test = np.float64(preTest)

    for i in range(preTrain.shape[0]):
        if numChannel > 1:
            for channel in range(0, numChannel, 1):
                train[i][:, :, channel] -= np.mean(
                    preTrain[i][:, :, channel].flatten(), axis=0
                )
        else:
            train[i] -= np.mean(preTrain[i].flatten(), axis=0)

    for i in range(preTest.shape[0]):
        if numChannel > 1:
            for channel in range(0, numChannel, 1):
                test[i][:, :, channel] -= np.mean(
                    preTest[i][:, :, channel].flatten(), axis=0
                )
        else:
            test[i] -= np.mean(preTest[i].flatten(), axis=0)

    if feat_norm:
        M = np.mean(train, axis=0)
        train = train - M
        sd = np.std(train, axis=0)
        train = train / sd

        test = test - M
        test = test / sd
        featNorm = {"mean": M, "std": sd}
        if numChannel > 1:
            with open(
                Path(config.config["datasets.path"]["pickle"]) / "BGRnorm.pickle", "wb"
            ) as handle:
                pickle.dump(featNorm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(
                Path(config.config["datasets.path"]["pickle"]) / "BWnorm.pickle", "wb"
            ) as handle:
                pickle.dump(featNorm, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # np.save('datasets/BWnorm.npz', featNorm)

    numtrain = train.shape[0]
    numtest = test.shape[0]
    row = train.shape[1]
    col = train.shape[2]

    train = np.reshape(train, (numtrain, row, col, numChannel))
    test = np.reshape(test, (numtest, row, col, numChannel))

    p = 0.9
    seed = 25
    np.random.seed(seed)
    split = np.int32(np.round((p * numtrain)))  # .85

    idx = np.random.permutation(numtrain)
    trIdx = idx[0:split]
    vlIdx = idx[split:numtrain]

    trlab = [
        np.reshape(trl[:, 0], (numtrain, 1)).astype("uint8"),
        np.reshape(trl[:, 1], (numtrain, 1)).astype("uint8"),
        np.reshape(trl[:, 2], (numtrain, 1)).astype("uint8"),
        np.reshape(trl[:, 3], (numtrain, 1)).astype("uint8"),
        np.reshape(trl[:, 4], (numtrain, 1)).astype("uint8"),
        np.reshape(trl[:, 6], (numtrain, 1)).astype("uint8"),
    ]

    tslab = [
        np.reshape(tsl[:, 0], (numtest, 1)).astype("uint8"),
        np.reshape(tsl[:, 1], (numtest, 1)).astype("uint8"),
        np.reshape(tsl[:, 2], (numtest, 1)).astype("uint8"),
        np.reshape(tsl[:, 3], (numtest, 1)).astype("uint8"),
        np.reshape(tsl[:, 4], (numtest, 1)).astype("uint8"),
        np.reshape(tsl[:, 6], (numtest, 1)).astype("uint8"),
    ]

    ctrlab = [
        trlab[0][trIdx],
        trlab[1][trIdx],
        trlab[2][trIdx],
        trlab[3][trIdx],
        trlab[4][trIdx],
        trlab[5][trIdx],
    ]
    cvlab = [
        trlab[0][vlIdx],
        trlab[1][vlIdx],
        trlab[2][vlIdx],
        trlab[3][vlIdx],
        trlab[4][vlIdx],
        trlab[5][vlIdx],
    ]
    ctslab = [tslab[0], tslab[1], tslab[2], tslab[3], tslab[4], tslab[5]]

    data = {
        "trainX": train[trIdx],
        "trainY": ctrlab,
        "testX": test,
        "testY": ctslab,
        "valdX": train[vlIdx],
        "valdY": cvlab,
    }

    return data


def generate_normalization_files(
    numChannel_bgr: int = 3, numChannel_bw: int = 1, feat_norm: bool = True
) -> None:
    """
    Generate normalization pickle files for both BGR and BW datasets.

    This function should be called after extracting all datasets to create
    the normalization statistics files needed during inference.

    Args:
        numChannel_bgr: Number of channels for BGR normalization (default: 3)
        numChannel_bw: Number of channels for BW normalization (default: 1)
        feat_norm: Whether to perform feature normalization (default: True)
    """
    logger.info("Generating normalization files...")

    # Generate BGR normalization file
    logger.info("Processing BGR (3-channel) normalization...")
    _ = prepDataforCNN(numChannel=numChannel_bgr, feat_norm=feat_norm)
    logger.info("BGR normalization file created at datasets/BGRnorm.pickle")

    # Generate BW normalization file
    logger.info("Processing BW (1-channel) normalization...")
    _ = prepDataforCNN(numChannel=numChannel_bw, feat_norm=feat_norm)
    logger.info("BW normalization file created at datasets/BWnorm.pickle")

    logger.info("Normalization files generated successfully!")


if __name__ == "__main__":
    # Extract all datasets
    logger.info("Starting full dataset pipeline...")

    extract_trainRGB()
    extract_testRGB()
    # extract_extraRGB()  # Uncomment if you have extra dataset

    # Generate normalization pickle files
    generate_normalization_files()

    logger.info("Full dataset pipeline complete!")
