from src.database import DatabaseManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class Augmenter:
    # TODO: Implement this class
    # Will need to take an input directory, which will be the result of
    # downloading from the ls client, and then augment each image and label
    # for things such as rotation, contrast/saturation/exposure etc.
    # Likey a threadpool implementation
    pass
