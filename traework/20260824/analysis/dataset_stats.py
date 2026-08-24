import json
from pathlib import Path

root = Path(r"D:\Data\New_Codes\Python_Codes\ultralytics\datasets\SSDC-UAV_coco\annotations")

for split in ["train", "val", "test"]:
    p = root / f"{split}.json"
    if not p.exists():
        print(f"{split}: MISSING")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    imgs = d.get("images", [])
    anns = d.get("annotations", [])
    cats = d.get("categories", [])
    n = len(anns)
    small = sum(1 for a in anns if a["area"] < 32 ** 2)
    medium = sum(1 for a in anns if 32 ** 2 <= a["area"] < 96 ** 2)
    large = sum(1 for a in anns if a["area"] >= 96 ** 2)
    areas = [a["area"] for a in anns]
    ws = [a["bbox"][2] for a in anns]
    hs = [a["bbox"][3] for a in anns]
    wh = sorted(ws)[len(ws) // 2], sorted(hs)[len(hs) // 2]
    imsz = [(i.get("width"), i.get("height")) for i in imgs[:5]]
    per_img = n / max(len(imgs), 1)
    print(
        f"{split}: images={len(imgs)} instances={n} cats={[c['name'] for c in cats]} "
        f"small(<32px^2)={small}({small / max(n,1) * 100:.1f}%) "
        f"medium={medium}({medium / max(n,1) * 100:.1f}%) "
        f"large={large}({large / max(n,1) * 100:.1f}%) "
        f"median_box=({wh[0]:.1f}x{wh[1]:.1f}px) mean_area={sum(areas)/max(n,1):.0f} "
        f"anns/img={per_img:.1f} sample_img_sz={imsz}"
    )
