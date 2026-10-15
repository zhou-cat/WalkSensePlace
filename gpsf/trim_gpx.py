"""
trim_gpx.py  --  trim walk GPX files to a step range that matches build_all_walks.py's
0-based timeline. Non-destructive: writes <name>_trim.gpx, never edits the original or any
insole/training CSV.

MODEL OF "STEP": build_all_walks draws one map segment per consecutive pair of timestamped
GPX trackpoints, and labels them 0-based. So:
   step k  <-> segment index k  <-> the pair (trackpoint[k], trackpoint[k+1])
To KEEP steps first..last (inclusive, 0-based), we keep trackpoints[first .. last+1] inclusive
(that's last-first+1 steps, last-first+2 points). Trimming the tail also trims the drawn route,
since the whole map is derived from the trackpoint list.

EDIT the TRIMS table below (recording stem -> (first_step, last_step), 0-based inclusive).
Run in the folder with the GPX files:   python trim_gpx.py
"""
import os, glob, re, sys

# recording stem (as it appears in the GPX filename) -> (first_step, last_step)   [0-based inclusive]
TRIMS = {
    '161001': (0, 395),
    '155900': (0, 362),
    '155015': (0, 294),
    '154547': (0, 106),
    '180315': (0, 239),
    '175445': (0, 308),
    '174713': (0, 246),
    '174319': (0, 119),
    '173910': (0, 104),
    '190906': (2, 409),   # drop steps 0-1; old step 2 becomes new step 0
}

def trkpt_blocks(raw):
    # returns (header, [ (block_text, has_time, time_str) ... ], footer) preserving exact XML
    # locate the <trkseg> ... </trkseg> body and split trkpts inside it
    return re.findall(r'<trkpt\b.*?</trkpt>\s*', raw, re.S)

def trim_file(path, first, last):
    raw = open(path, encoding='utf-8').read()
    blocks = re.findall(r'<trkpt\b.*?</trkpt>\s*', raw, re.S)
    # keep only timestamped trackpoints, exactly like build_all_walks' parse_gpx
    timed = [(b, re.search(r'<time>([^<]+)</time>', b)) for b in blocks]
    timed_blocks = [b for (b, m) in timed if m]
    n = len(timed_blocks)
    # keep points [first .. last+1] inclusive  (0-based step k uses points k, k+1)
    lo = first
    hi = min(last + 1, n - 1)
    if lo < 0 or lo > hi:
        print(f"  !! {os.path.basename(path)}: invalid range first={first} last={last} (n_points={n})"); return None
    kept = timed_blocks[lo:hi+1]
    kept_steps = len(kept) - 1
    # rebuild file: replace the full run of original <trkpt> blocks with the kept ones
    # find span from first <trkpt> to last </trkpt>
    first_i = raw.find('<trkpt')
    last_i = raw.rfind('</trkpt>') + len('</trkpt>')
    new_body = ''.join(kept).rstrip()
    new_raw = raw[:first_i] + new_body + '\n      ' + raw[last_i:]
    out = os.path.splitext(path)[0] + '_trim.gpx'
    open(out, 'w', encoding='utf-8').write(new_raw)
    t0 = re.search(r'<time>([^<]+)</time>', kept[0]).group(1)
    t1 = re.search(r'<time>([^<]+)</time>', kept[-1]).group(1)
    print(f"  {os.path.basename(path):34} points {n} -> {len(kept)}  (steps {first}..{last} = {kept_steps} steps)")
    print(f"       first kept time {t0}   last kept time {t1}")
    return out

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    gpx = sorted(glob.glob(os.path.join(folder, '*.gpx')))
    gpx = [g for g in gpx if not g.endswith('_trim.gpx')]
    print(f"{len(gpx)} GPX file(s) in {folder}\n")
    done = 0
    for path in gpx:
        base = os.path.basename(path)
        stem = next((k for k in TRIMS if k in base), None)
        if stem is None:
            print(f"  {base:34} (no trim entry — skipped)"); continue
        first, last = TRIMS[stem]
        if trim_file(path, first, last): done += 1
    print(f"\nwrote {done} trimmed file(s) as *_trim.gpx (originals untouched).")
    print("Point build_all_walks.py at the trimmed GPX (or replace originals once verified).")

if __name__ == '__main__':
    main()
