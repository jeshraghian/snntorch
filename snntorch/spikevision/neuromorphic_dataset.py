import os
import torch
import torch.utils.data as data
from torchvision.datasets.utils import (
    verify_str_arg,
    check_integrity,
    gen_bar_updater,
)
from ._transforms import Compose
from tqdm import tqdm
import tarfile, zipfile, gzip


# Adapted from https://github.com/nmi-lab/torchneuromorphic by Emre Neftci and Clemens Schaefer

DEFAULT_ROOT = "data/"


def download_url(url, root, filename=None, md5=None, total_size=None):

    """Download a file from a url and place it in root.

    :param url: URL to download file from
    :type url: string

    :param root: Directory to place downloaded file in
    :type root: string

    :param filename: Name to save the file under. If None, use the basename of the URL
    :type filename: string, optional

    :param md5: MD5 checksum of the download. If None, do not check
    :type md5: string, optional
    """

    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:  # download the file
        try:
            if "dropbox" in url:
                # Handle dropbox links differently
                import requests

                headers = {"user-agent": "Wget/1.16 (linux-gnu)"}
                r = requests.get(url, stream=True, headers=headers)
                # # new
                total_size_in_bytes = int(r.headers.get("content-length", 0))  # new
                block_size = 1024  # 1 Kibibyte - new
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )  # new
                # old
                with open(fpath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            progress_bar.update(len(chunk))  # new
                            f.write(chunk)
                progress_bar.close()  # new
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print(
                        "Warning: Downloaded size {progress_bar.n} does not match {total_size_in_bytes}."
                    )

            elif "Manual" in url:
                raise urllib.error.URLError(url)
            else:
                print("Downloading " + url + " to " + fpath)
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def download_and_extract_archive(
    url,
    download_root,
    extract_root=None,
    filename=None,
    md5=None,
    remove_finished=False,
):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))  # test *might need to change syntax to colab-friendly here
    _extract_archive(archive, extract_root, remove_finished)


# backwards compatibility
def identity(x):
    return x


class NeuromorphicDataset(data.Dataset):
    _repr_indent = 4

    def __init__(
        self, root=None, transforms=None, transform=None, target_transform=None, transform_train=None, transform_test=None, target_transform_train=None, target_transform_test=None,
    ):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        if root is not None:
            if not os.path.isfile(root):
                if self.download_and_create:
                    self._download()  # check if raw files exist in root
                    self._create_hdf5()  #
                else:
                    raise Exception(
                        "File {} does not exist and download_and_create is False".format(
                            root
                        )
                    )

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can "
                "be passed as argument"
            )

        # for backwards-compatibility
        if transform is None:
            transform = identity
        if target_transform is None:
            target_transform = identity

        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def extra_repr(self):
        return ""

    def _check_exists(self):
        res_ = [os.path.exists(d) for d in self.resources_local]
        res = all(res_)
        if res is False:
            print("The following files do not exist, will attempt download:")
            for i, r in enumerate(res_):
                if not r:
                    print(self.resources_local[i])
        return res

    def _extract_exists(self):
        res_ = [os.path.exists(d) for d in self.resources_local_extracted]
        res = all(res_)
        if res is False:
            for _, _, filename in self._resources_url:
                extract_root = self.directory
                archive = os.path.join(extract_root, filename).replace('\\', '/')
                print("Extracting {} to {}...".format(archive, extract_root))  # test?
                _extract_archive(archive, extract_root, remove_finished=False)
        return res

    def _download(self):
        if self._check_exists():
            if self._extract_exists():  # test
                return True
        else:
            os.makedirs(self.directory, exist_ok=True)
            for url, md5, filename in self._resources_url:
                download_and_extract_archive(
                    url, download_root=self.directory, filename=filename, md5=md5
                )
            return False

    def _create_hdf5(self):
        raise NotImplementedError()

    def target_transform_append(self, transform):
        if transform is None:
            return
        if self.target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = Compose([self.target_transform, transform])

    def transform_append(self, transform):
        if transform is None:
            return
        if self.transform is None:
            self.transform = transform
        else:
            self.transform = Compose([self.transform, transform])

def _extract_archive(from_path, to_path = None, remove_finished = False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        if _is_tar(from_path):
            with tarfile.open(from_path, 'r') as tar:
                tar.extractall(path=to_path)
        elif _is_targz(from_path) or _is_tgz(from_path):
            with tarfile.open(from_path, 'r:gz') as tar:
                tar.extractall(path=to_path)
        elif _is_tarxz(from_path):
            with tarfile.open(from_path, 'r:xz') as tar:
                tar.extractall(path=to_path)
        elif _is_gzip(from_path):
            to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
            with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
                out_f.write(zip_f.read())
        elif _is_zip(from_path):
            with zipfile.ZipFile(from_path, 'r') as z:
                z.extractall(to_path)
        else:
            raise ValueError("Extraction of {} not supported".format(from_path))

        if remove_finished:
            os.remove(from_path)

class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(
                self.target_transform, "Target transform: "
            )

        return "\n".join(body)

def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")


def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
