"""Index portable exported records offline; never decode SVO or invoke an SDK."""
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import stat
import tempfile

from . import schema

EXPORT_SCHEMA = "stereoforge.zed_comparison.export.v1"
INDEX_SCHEMA = "stereoforge.zed_comparison.index.v1"


def _fingerprint(path):
    try:
        info = path.lstat()
        schema._require(stat.S_ISREG(info.st_mode), "manifest must be a regular file")
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    except OSError as exc:
        raise ValueError("cannot stat export manifest") from exc


def _unchanged(path, before):
    schema._require(_fingerprint(path) == before, "export manifest changed during indexing")


def _existing(path, data):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    schema._require(stat.S_ISREG(info.st_mode), "index must be a regular file")
    schema._require(info.st_size == len(data) and path.read_bytes() == data,
                    "different index already exists")
    return True


def index_export(root: Path) -> Path:
    """Validate a sealed portable export and atomically publish its unqualified index."""
    root = Path(root)
    schema._require(".." not in root.parts, "unsafe export root")
    root = root.absolute()
    for component in (root, *root.parents):
        schema._require(not component.is_symlink(), "symlink export root forbidden")
    manifest_path = root / "export.json"
    before = _fingerprint(manifest_path)
    manifest = schema._read_json(manifest_path)
    _unchanged(manifest_path, before)
    schema._keys(manifest, {"schema", "records"}, "export")
    schema._require(manifest["schema"] == EXPORT_SCHEMA, "unknown export schema")
    descriptors = manifest["records"]
    schema._require(type(descriptors) is list and bool(descriptors), "empty export records")
    names = []
    for desc in descriptors:
        schema._descriptor(desc)
        name = desc["path"]
        schema._require("/" not in name and name.endswith(".json")
                        and name not in {"export.json", "index.json"}, "unsafe record basename")
        names.append(name)
    schema._require(len(set(names)) == len(names), "duplicate export record path")
    digest = sha256()
    size = 0
    with manifest_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    _unchanged(manifest_path, before)
    records = schema.load_records([root / name for name in names], expected=descriptors)
    _unchanged(manifest_path, before)
    groups = {}
    for name, record in zip(names, records):
        groups.setdefault(record["input_identity"], []).append(name)
    index = dict(schema=INDEX_SCHEMA,
                 export=dict(path="export.json", sha256=digest.hexdigest(), size_bytes=size),
                 records=descriptors,
                 groups=[dict(input_identity=identity, records=sorted(groups[identity]))
                         for identity in sorted(groups)],
                 status="indexed", quality_status="unqualified")
    data = (json.dumps(index, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    target = root / "index.json"
    if _existing(target, data):
        return target
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".index-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            _existing(target, data)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(index_export(args.root))


if __name__ == "__main__":
    main()
