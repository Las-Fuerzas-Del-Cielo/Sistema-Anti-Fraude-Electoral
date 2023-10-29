import os
from pathlib import Path
from typing import *


def check_dir(directory: Union[str, Path], exist: bool = True):
    directory = Path(directory)

    if exist:
        assert directory.is_dir(), f"Directory {directory.resolve()} does not exist!"
    else:
        assert directory.parent.is_dir(), f"Parent directory {directory.parent.resolve()} does not exist!"
        assert not directory.is_dir(), f"Directory {directory.resolve()} does exist!"

    return directory


def check_file(file: Union[str, Path], suffix: Union[None, str, List[str]] = None, exist: Optional[bool] = True):
    file = Path(file)

    if exist is None:
        pass  # No check
    elif exist:
        assert file.is_file(), f"File {file.resolve()} does not exist!"
    else:
        assert file.parent.is_dir(), f"Parent directory {file.parent.resolve()} does not exist!"
        assert not file.is_file(), f"File {file.resolve()} does exist!"

    if suffix is None:
        pass
    elif isinstance(suffix, str):
        assert file.suffix == suffix, f"File {file.resolve()} has an invalid suffix! Allowed is '{suffix}'"
    else:
        assert file.suffix in suffix, f"File {file.resolve()} has an invalid suffix! Allowed is any of '{suffix}'"

    return file


def list_dirs(search_dir: Union[str, Path], recursive: bool = False, as_string: bool = False, glob_string="*"):
    search_dir = Path(search_dir)
    glob_function = search_dir.rglob if recursive else search_dir.glob

    dirs = [str(file) if as_string else file
            for file in glob_function(glob_string)
            if file.is_dir()]
    return list(sorted(dirs))


def list_files(search_dir: Union[str, Path], suffixes: List[str] = None, recursive: bool = False,
               as_string: bool = False):
    search_dir = check_dir(search_dir)

    if suffixes is None:
        suffixes = ['']

    glob_function = search_dir.rglob if recursive else search_dir.glob

    files = [str(file) if as_string else file
             for suffix in suffixes
             for file in glob_function("*" + suffix)
             if file.is_file()]

    return list(sorted(files))


def is_empty(directory: Union[str, Path]) -> bool:
    directory = Path(directory)
    return not any(directory.iterdir())


def remove_common_path(path: Union[str, Path], reference: Union[str, Path]) -> Path:
    path = Path(path)
    reference = Path(reference)

    path = path.expanduser().absolute()
    reference = reference.expanduser().absolute()
    common_path = os.path.commonpath([str(path), str(reference)])
    return path.relative_to(Path(common_path))
