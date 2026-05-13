import sys
import logging
from pathlib import Path
from .utils import print_distribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data(pathname):
    data = {}
    if not pathname.is_dir():
        logger.error("Path folder is not a directory.")
        exit(1)
    for subdir in pathname.iterdir():
        if subdir.is_dir():
            # compte tous les JPG dans le sous-dossier et ses
            # sous-sous-dossiers
            jpg_files = list(subdir.rglob("*.JPG"))
            data[subdir.name] = len(jpg_files)
    return data


def main():
    if len(sys.argv) != 2:
        logger.error("Path folder is missing.")
        exit(1)
    pathname = Path(sys.argv[1])
    data = get_data(pathname)
    print_distribution(data)


if __name__ == "__main__":
    main()
