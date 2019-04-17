"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import os
import logging
import shutil
import tempfile
import json
# from urlparse import urlparse
from pathlib import Path
# from typing import Optional, Tuple, Union, IO, Callable, Set
# from hashlib import sha256
# from functools import wraps

# from tqdm import tqdm

# import boto3
# from botocore.exceptions import ClientError
# import requests

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

PYTORCH_PRETRAINED_BERT_CACHE = ''
# PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
#                                                Path.home() / '.pytorch_pretrained_bert'))


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def cached_path(url_or_filename, cache_dir = None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    return url_or_filename


# def get_from_cache(url, cache_dir=None):
#     """
#     Given a URL, look for the corresponding dataset in the local cache.
#     If it's not there, download it. Then return the path to the cached file.
#     """
#     if cache_dir is None:
#         cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
#     if isinstance(cache_dir, Path):
#         cache_dir = str(cache_dir)

#     os.makedirs(cache_dir, exist_ok=True)

#     # Get eTag to add to filename, if it exists.
#     if url.startswith("s3://"):
#         etag = s3_etag(url)
#     else:
#         response = requests.head(url, allow_redirects=True)
#         if response.status_code != 200:
#             raise IOError("HEAD request failed for url {} with status code {}"
#                           .format(url, response.status_code))
#         etag = response.headers.get("ETag")

#     filename = url_to_filename(url, etag)

#     # get cache path to put the file
#     cache_path = os.path.join(cache_dir, filename)

#     if not os.path.exists(cache_path):
#         # Download to temporary file, then copy to cache dir once finished.
#         # Otherwise you get corrupt cache entries if the download gets interrupted.
#         with tempfile.NamedTemporaryFile() as temp_file:
#             logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

#             # GET file object
#             if url.startswith("s3://"):
#                 s3_get(url, temp_file)
#             else:
#                 http_get(url, temp_file)

#             # we are copying the file before closing it, so flush to avoid truncation
#             temp_file.flush()
#             # shutil.copyfileobj() starts at the current position, so go to the start
#             temp_file.seek(0)

#             logger.info("copying %s to cache at %s", temp_file.name, cache_path)
#             with open(cache_path, 'wb') as cache_file:
#                 shutil.copyfileobj(temp_file, cache_file)

#             logger.info("creating metadata file for %s", cache_path)
#             meta = {'url': url, 'etag': etag}
#             meta_path = cache_path + '.json'
#             with open(meta_path, 'w') as meta_file:
#                 json.dump(meta, meta_file)

#             logger.info("removing temp file %s", temp_file.name)

#     return cache_path


def read_set_from_file(filename):
    '''
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    '''
    collection = set()
    with open(filename, 'r', encoding='utf-8') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path, dot=True, lower=True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext
