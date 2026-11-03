#!/usr/bin/env python3
"""Verify release integrity only; kernel migration and native checks are separate."""
import argparse, hashlib, json, pathlib, re, subprocess, tarfile, tempfile, urllib.request
REPO = "bjoernellens1/sintergpu"
WORKFLOW = REPO + "/.github/workflows/release.yml"
def require(value, message):
    if not value: raise ValueError(message)
def gh(*args):
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout
def main():
    p = argparse.ArgumentParser(); p.add_argument("--lock", required=True); args = p.parse_args()
    lock = json.loads(pathlib.Path(args.lock).read_text())
    require(lock.get("schema_version") == 1 and lock.get("repository") == REPO, "schema/repository")
    require(re.fullmatch(r"v(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)", lock["tag"]) and lock["tag"] == "v"+lock["version"], "version/tag")
    require(re.fullmatch("[a-f0-9]{40}", lock["source_commit"]), "source commit")
    info = json.loads(gh("release", "view", lock["tag"], "-R", REPO, "--json", "isDraft,isImmutable"))
    require(not info["isDraft"] and info.get("isImmutable"), "immutable publication required")
    gh("release", "verify", lock["tag"], "-R", REPO)
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for key in ("manifest", "artifact"):
            pin = lock[key]; name = pin["name"]
            require(re.fullmatch("[A-Za-z0-9_.+-]+", name) and name not in (".", ".."), "unsafe asset path")
            require(pin["url"] == "https://github.com/"+REPO+"/releases/download/"+lock["tag"]+"/"+name, "asset URL")
            target = pathlib.Path(tmp)/name
            with urllib.request.urlopen(pin["url"], timeout=60) as response: target.write_bytes(response.read())
            require(hashlib.sha256(target.read_bytes()).hexdigest() == pin["sha256"], "asset hash")
            gh("release", "verify-asset", lock["tag"], str(target), "-R", REPO)
            gh("attestation", "verify", str(target), "--repo", REPO, "--signer-workflow", WORKFLOW, "--source-ref", "refs/tags/"+lock["tag"], "--source-digest", lock["source_commit"], "--deny-self-hosted-runners")
            paths[key] = target
        manifest = json.loads(paths["manifest"].read_text())
        for key in ("repository", "version", "tag", "source_commit", "shader_tree_sha256"):
            require(manifest.get(key) == lock[key], "manifest mismatch: "+key)
        require(manifest.get("workflow") == WORKFLOW, "workflow mismatch")
        require(any(a["name"] == lock["artifact"]["name"] and a["sha256"] == lock["artifact"]["sha256"] and a["size"] == paths["artifact"].stat().st_size for a in manifest["assets"]), "artifact not bound by manifest")
        with tarfile.open(paths["artifact"]) as archive:
            for member in archive:
                path = pathlib.PurePosixPath(member.name)
                require(not path.is_absolute() and ".." not in path.parts and not member.issym() and not member.islnk() and (member.isfile() or member.isdir()), "unsafe archive member")
    print("Release integrity verified; consumer/native qualification remains separate.")
if __name__ == "__main__": main()
