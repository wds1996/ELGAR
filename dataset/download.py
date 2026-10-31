"""Download or import official processed data into dataset/<source>/<name>/."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
import zipfile
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from elgar.utils import file_digest, write_json
from .preprocess import ALIASES, canonical_name, validate_native_pickle, validate_interactions

# RecBole's official Amazon_ratings folder, linked by the ReAd README.
RECBOLE_FILE_IDS = {
    'Amazon_Office': '1NBIx7culdexQUeF0cR9hNrMZ-8ZiHEDQ',
    'Amazon_Beauty': '16BDe6IT0mQeBK519tEPSgofdRF2KwgPV',
    'Amazon_Sports_and_Outdoors': '1VkC55X9NTOsnwApFYEZvTQ0807bLUzlZ',
    'Amazon_Home': '1tHwVPIkGxi_LWk4NMCYmhdg2eVstoHh9',
}
FAVE_REVISION = '09e81fb7c2fdfaf54092f7d5227ac085b78cc15d'


def interaction_payload(path, name, temporary_root):
    """Read the matching interaction file from a RecBole ZIP, if necessary.

    Stream a single member to a fixed destination; never extract archive paths.
    """
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return path
    normalize = lambda text: re.sub(r'[^a-z0-9]', '', text.lower())
    names = {normalize(name)} | {normalize(alias) for alias, target in ALIASES.items()
                                if target == name}
    with zipfile.ZipFile(path) as archive:
        candidates = [entry for entry in archive.infolist()
                      if not entry.is_dir() and Path(entry.filename).suffix == '.inter'
                      and normalize(Path(entry.filename).stem) in names]
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one matching {name} .inter file in {path}.')
        output = Path(temporary_root) / f'{name}.inter'
        with archive.open(candidates[0]) as incoming, output.open('wb') as handle:
            shutil.copyfileobj(incoming, handle)
    return output


def install_file(source, destination, *, overwrite=False, sha256=None):
    """Validate and install a file atomically, retaining the original bytes."""
    source, destination = Path(source), Path(destination)
    digest = file_digest(source)
    if sha256 is not None and digest.lower() != sha256.lower():
        raise ValueError(f'SHA256 mismatch for {source}')
    if destination.suffix == '.inter':
        validate_interactions(source)
    else:
        validate_native_pickle(source)
    if destination.is_file():
        if file_digest(destination) == digest:
            return digest
        if not overwrite:
            raise FileExistsError(f'{destination} differs; pass --overwrite to replace it.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with source.open('rb') as incoming:
                shutil.copyfileobj(incoming, handle)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return digest


def fetch(url, destination):
    """Fetch a single file; Google Drive file links use optional gdown."""
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError('Use an HTTP(S) file URL, or --local-file for local data.')
    if parsed.hostname == 'drive.google.com':
        if '/folders/' in parsed.path:
            raise ValueError('Pass a single Google Drive file link, not a folder link.')
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError("Google Drive links require: pip install -e '.[data]'") from exc
        # gdown 6 removed fuzzy=. Passing a parsed file ID works with both
        # the supported 5.x releases and current 6.x releases.
        ids = parse_qs(parsed.query).get('id')
        match = re.search(r'/file/d/([^/]+)', parsed.path)
        file_id = ids[0] if ids else match.group(1) if match else None
        if not file_id:
            raise ValueError('Expected an individual Google Drive file link with a file ID.')
        if not gdown.download(id=file_id, output=str(destination), quiet=False):
            raise RuntimeError('Google Drive download did not return a file.')
    else:
        request = Request(url, headers={'User-Agent': 'ELGAR-data-preparation'})
        with urlopen(request, timeout=60) as response, Path(destination).open('wb') as handle:
            shutil.copyfileobj(response, handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('recbole', 'fave'), required=True)
    parser.add_argument('--name', required=True, help='Paper name or canonical dataset identifier.')
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument('--url', help='Direct file or Google Drive link; RecBole ZIP archives are supported.')
    origin.add_argument('--local-file', type=Path, help='Import an already downloaded file.')
    origin.add_argument('--fave-root', type=Path, help='Import from an official FAVE checkout.')
    parser.add_argument('--revision', default=FAVE_REVISION, help='Official FAVE Git revision for automatic download.')
    parser.add_argument('--root', type=Path, default=Path('dataset'))
    parser.add_argument('--sha256', help='Expected SHA256 of the installed .inter/.pkl file (after ZIP extraction).')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    name = canonical_name(args.source, args.name)
    if args.fave_root and args.source != 'fave':
        parser.error('--fave-root applies only to FAVE data.')
    filename = f'{name}.inter' if args.source == 'recbole' else 'dataset.pkl'
    destination = args.root / args.source / name / filename
    url = args.url
    local = args.local_file
    if args.fave_root:
        local = args.fave_root / 'datasets/data' / name / filename
    elif local is None and url is None:
        if args.source == 'recbole':
            url = f'https://drive.google.com/file/d/{RECBOLE_FILE_IDS[name]}/view'
        else:
            url = (f'https://raw.githubusercontent.com/Blue130/Fave/{args.revision}'
                   f'/datasets/data/{name}/dataset.pkl')
    metadata_path = destination.parent / 'download.json'
    if url and destination.is_file() and metadata_path.is_file() and not args.overwrite:
        old = json.loads(metadata_path.read_text())
        digest = file_digest(destination)
        if old.get('origin') == url and old.get('sha256') == digest:
            if args.sha256 is not None and args.sha256.lower() != digest:
                raise ValueError('Existing file does not match --sha256.')
            print(f'[reuse] {destination} ({digest})')
            return destination
    if local is not None:
        with tempfile.TemporaryDirectory(prefix='elgar-unpack-') as tmp:
            payload = interaction_payload(local, name, tmp) if args.source == 'recbole' else local
            digest = install_file(payload, destination, overwrite=args.overwrite, sha256=args.sha256)
        container_digest = file_digest(local)
        origin_description = str(local.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='elgar-download-') as tmp:
            incoming = Path(tmp) / 'downloaded_file'
            fetch(url, incoming)
            container_digest = file_digest(incoming)
            payload = interaction_payload(incoming, name, tmp) if args.source == 'recbole' else incoming
            digest = install_file(payload, destination, overwrite=args.overwrite, sha256=args.sha256)
        origin_description = url
    write_json(metadata_path, {'source': args.source, 'dataset': name,
                               'origin': origin_description, 'container_sha256': container_digest,
                               'sha256': digest})
    print(f'[ready] {destination} ({digest})')
    return destination


if __name__ == '__main__':
    main()
