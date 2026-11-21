#!/usr/bin/env python3
"""Verify all cached RGB pixels against pinned Clean videos before training.

Uses exactly the same bounded ffmpeg decoder, lock, comparison, and per-view
receipt as training's first read. Existing mismatching pixels are never changed.
Missing views are generated from the verified source. This optional CPU job
reads roughly 730 GB of decoded pixels for the full current Clean export.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinkwam_p1.robotwin_protocol import validate_clean_rgb_cache
from thinkwam_p1.data.robotwin_online import ensure_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='pinned Clean LeRobot dataset root')
    parser.add_argument('--frame-cache', required=True)
    parser.add_argument('--manifest-name', default='manifest_3view.json')
    args = parser.parse_args()
    evidence = validate_clean_rgb_cache(args.root, frame_cache=args.frame_cache,
                                        manifest_name=args.manifest_name)
    cache = Path(args.frame_cache).resolve()
    manifest = json.loads((cache / args.manifest_name).read_text(encoding='utf-8'))
    verified = 0
    for episode in manifest['episodes']:
        for view in range(3):
            ensure_frames(cache, episode, view, 384, source_root=Path(evidence['source_root']),
                          source_files=evidence['source_files'])
            verified += 1
        print(json.dumps(dict(verified_views=verified, planned_views=len(manifest['episodes']) * 3)), flush=True)
    print(json.dumps(dict(verified_views=verified, all_pixels_verified=True)), flush=True)


if __name__ == '__main__':
    main()
