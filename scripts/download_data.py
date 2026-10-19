"""公开医学影像数据集下载助手。

收录的数据集链接均于 2026-10-19 逐个核实可访问。
多数数据集托管在 Kaggle，需要：
  1. 注册 Kaggle 账号；
  2. 在账号设置页下载 kaggle.json（含 API Key）；
  3. 把 kaggle.json 放到 C:\\Users\\<你的用户名>\\.kaggle\\kaggle.json（Windows）
     或 ~/.kaggle/kaggle.json（Linux/Mac）；
  4. pip install kaggle

用法：
    python scripts/download_data.py list                 # 列出所有数据集
    python scripts/download_data.py download xray_pneumonia
    python scripts/download_data.py download ct_lung --dest data/raw

非 Kaggle 来源（NIH 官网、TCIA）需浏览器手动下载，见 技术文档 docs/技术文档.md。
"""
import argparse
import os
import shutil
import subprocess
import sys

# key: (名称, kaggle slug 或 None, 官网链接, 规模, 类别, 备注)
DATASETS = {
    "xray_pneumonia": {
        "name": "Chest X-Ray Images (Pneumonia) 胸部X光肺炎二分类",
        "kaggle": "paultimothymooney/chest-xray-pneumonia",
        "url": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        "size": "约 1.2 GB / 5,856 张（已验证）",
        "classes": "NORMAL / PNEUMONIA",
        "note": "经典入门数据集。官方 val 仅 16 张，务必用 scripts/split_dataset.py "
                "从 train 重新划分；test 留给最终评估。",
        "layout": "解压后 chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg",
    },
    "xray_pneumonia_balanced": {
        "name": "Pneumonia X-Ray Images（重新平衡划分版）",
        "kaggle": "pcbreviglieri/pneumonia-xray-images",
        "url": "https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images",
        "size": "约 1.24 GB / 5,856 张（已验证）",
        "classes": "normal / opacity",
        "note": "上一个数据集的重划分版本，train 4192 / val 1040 / test 624，"
                "val 足够大，推荐直接使用。",
        "layout": "解压后 {train,val,test}/{normal,opacity}/*.jpeg",
    },
    "ct_lung": {
        "name": "Chest CT-Scan Images 胸部CT肺癌四分类",
        "kaggle": "mohamedhanyyy/chest-ctscan-images",
        "url": "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images",
        "size": "数百 MB / 1,000 张（规模经文献核实，体积未精确核实）",
        "classes": "adenocarcinoma / large_cell_carcinoma / squamous_cell_carcinoma / normal",
        "note": "小规模 CT 数据集，适合 8GB 显存快速实验。",
        "layout": "解压后 Data/{train,valid,test}/<类别>/*.png（注意目录名是 valid 不是 val，"
                  "用 --data-root .../Data 即可）",
    },
    "mri_brain": {
        "name": "Brain MRI Images for Brain Tumor Detection 脑部MRI二分类",
        "kaggle": "navoneel/brain-mri-images-for-brain-tumor-detection",
        "url": "https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection",
        "size": "约 20 MB / 253 张（规模已验证，体积为常见值）",
        "classes": "no / yes",
        "note": "最小数据集，适合先跑通全流程。解压后是扁平目录，需 split_dataset.py 划分。",
        "layout": "解压后 <flat>/{no,yes}/*.jpg → 需划分成 train/val/test",
    },
    "mri_brain4": {
        "name": "Brain Tumor Classification (MRI) 脑部MRI四分类",
        "kaggle": "sartajbhuvaji/brain-tumor-classification-mri",
        "url": "https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri",
        "size": "约 3,264 张（规模经 IEEE 论文核实）",
        "classes": "glioma / meningioma / pituitary / no_tumor",
        "note": "比上一个更适合正式实验，自带 train/test 划分。",
        "layout": "解压后 Training/<类别>/*.jpg 与 Testing/<类别>/*.jpg",
    },
    "nih_chestxray14": {
        "name": "NIH Chest X-ray 14 (224x224 重采样版)",
        "kaggle": "khanfashee/nih-chest-x-ray-14-224x224-resized",
        "url": "https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized",
        "size": "112,120 张（规模已验证；总体积数 GB 级）",
        "classes": "14 种病变 + No Finding（多标签，官方 CSV 给出）",
        "note": "大规模数据集。注意：它是多标签任务，本项目的 train.py 按单标签多分类训练，"
                "需要自行改造（BCEWithLogitsLoss + 多热标签）。仅建议有经验后使用。",
        "layout": "解压后 images/*.png + Data_Entry_2017.csv",
    },
    "mri_brain_hf": {
        "name": "Brain Tumor MRI Dataset（masoudnickparvar 原版，HF 镜像）脑部MRI四分类",
        "kaggle": "masoudnickparvar/brain-tumor-mri-dataset",
        "hf": "BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset",
        "url": "https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset",
        "size": "7,200 张（Training 5,600 + Testing 1,600，每类各 1,400/400——已核实）/ 约 330 MB",
        "classes": "glioma / meningioma / notumor / pituitary",
        "note": "Kaggle 原版需账号，此条走 hf-mirror.com 免登录直链（huggingface.co 直连不通时用）。"
                "四类完全均衡，适合正式实验。",
        "layout": "Training/<类别>/*.jpg、Testing/<类别>/*.jpg（已是 ImageFolder 布局）",
    },
    "mri_brain_figshare": {
        "name": "Brain Tumor Dataset（Cheng et al. 2017，figshare 直链）脑部MRI四分类",
        "kaggle": None,
        "url": "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427",
        "size": "共 3,064 张 / 4 个 zip 分卷（约 880 MB，CC BY 4.0）",
        "classes": "meningioma / glioma / pituitary / no_tumor",
        "note": "免登录直链，自带 MD5 校验。文献常见分布：meningioma 708、glioma 1426、"
                "pituitary 930、无肿瘤 500（需下载后核实）。只练手可只下第一卷（766 张）："
                "--file 1-766",
        "layout": "zip 内为编号目录，需按 cvind.mat 标签整理（见 docs/技术文档.md）",
        "direct": [
            {"url": "https://ndownloader.figshare.com/files/3381290",
             "name": "brainTumorDataPublic_1-766.zip", "size": "约 214 MB",
             "md5": "74b949ad33f042e6e103523091cd1428"},
            {"url": "https://ndownloader.figshare.com/files/3381296",
             "name": "brainTumorDataPublic_767-1532.zip", "size": "约 208 MB",
             "md5": "7e8a875500d2c8a346f270538e29890e"},
            {"url": "https://ndownloader.figshare.com/files/3381293",
             "name": "brainTumorDataPublic_1533-2298.zip", "size": "约 206 MB",
             "md5": "8227bf6080cb71f15a88be8d25c79ae7"},
            {"url": "https://ndownloader.figshare.com/files/3381302",
             "name": "brainTumorDataPublic_2299-3064.zip", "size": "约 221 MB",
             "md5": "b378a80d6174e5317d59eb28430c6652"},
            {"url": "https://ndownloader.figshare.com/files/51340418",
             "name": "README_2024.txt", "size": "3 KB",
             "md5": "4188f7c30957e9673c3611613071bd6e"},
            {"url": "https://ndownloader.figshare.com/files/7005344",
             "name": "cvind.mat", "size": "6 KB",
             "md5": "5ea82eb3212c9fcb84290ccf0beb486f"},
        ],
    },
}

# 非 Kaggle 官方来源（需浏览器手动下载）
MANUAL_SOURCES = [
    ("NIH ChestX-ray14 官网（NIH Clinical Center，Box 网盘）",
     "https://nihcc.app.box.com/v/ChestXray-NIHCC"),
    ("NIH Chest X-ray Google Cloud 公共存储桶",
     "https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest"),
    ("LIDC-IDRI 肺部CT（TCIA，需注册账号）",
     "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI"),
    ("LUNA16 肺结节CT（无需注册）",
     "https://luna16.grand-challenge.org/Home/"),
]


def check_kaggle_ready():
    if shutil.which("kaggle") is None:
        print("[ERROR] 未找到 kaggle CLI。请先: pip install kaggle")
        return False
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json):
        print(f"[ERROR] 未找到 {kaggle_json}")
        print("        请登录 Kaggle → 账号设置 → Create New API Token，"
              "把下载的 kaggle.json 放到上述路径。")
        return False
    return True


def cmd_list():
    print(f"{'key':<24} {'名称':<42} {'规模'}")
    print("-" * 110)
    for key, d in DATASETS.items():
        print(f"{key:<24} {d['name'][:40]:<42} {d['size']}")
    print("\n非 Kaggle 官方来源（需浏览器手动下载）：")
    for name, url in MANUAL_SOURCES:
        print(f"  - {name}\n    {url}")


def download_hf(repo_id, dest, endpoint="https://hf-mirror.com", max_workers=8):
    """从 HuggingFace 镜像站免登录下载数据集（多线程直链）。

    huggingface.co 直连在某些网络不可达，用 hf-mirror.com 镜像。
    通过 HF API 列出仓库文件，再逐文件并发下载 resolve 直链。
    """
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.parse import quote

    api = f"{endpoint}/api/datasets/{repo_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        info = json.load(r)
    files = [s["rfilename"] for s in info.get("siblings", [])
             if not s["rfilename"].startswith(".")]
    print(f"[INFO] HF 数据集 {repo_id}: {len(files)} 个文件")
    base = f"{endpoint}/datasets/{repo_id}/resolve/main/"

    def fetch(rel):
        target = os.path.join(dest, *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            return rel, True, ""
        last_err = ""
        for _ in range(3):
            try:
                urllib.request.urlretrieve(base + quote(rel), target)
                return rel, True, ""
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
        return rel, False, last_err

    ok, fail = 0, []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            rel, success, err = fut.result()
            if success:
                ok += 1
            else:
                fail.append((rel, err))
            if i % 500 == 0 or i == len(files):
                print(f"  进度 {i}/{len(files)}")
    print(f"[INFO] 下载完成: 成功 {ok}，失败 {len(fail)}")
    for rel, err in fail[:10]:
        print(f"  [FAIL] {rel}: {err}")
    if fail:
        raise SystemExit("有文件下载失败，重跑即可（已下载的会跳过）")


def download_direct(files, dest, file_filter=None):
    """从直链下载（urllib，免登录），带 MD5 校验。"""
    import hashlib
    import urllib.request

    os.makedirs(dest, exist_ok=True)
    for f in files:
        if file_filter and file_filter not in f["name"]:
            continue
        target = os.path.join(dest, f["name"])
        print(f"[INFO] 下载 {f['name']} ({f.get('size', '?')}) -> {target}")

        def reporthook(count, block_size, total_size, _name=f["name"]):
            if total_size > 0 and count % 100 == 0:
                pct = min(100.0, count * block_size * 100 / total_size)
                print(f"\r  {_name}: {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(f["url"], target, reporthook)
        print()
        if f.get("md5"):
            h = hashlib.md5()
            with open(target, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            ok = h.hexdigest() == f["md5"]
            print(f"[INFO] MD5 校验: {'通过' if ok else '不匹配！'}")
            if not ok:
                raise SystemExit(f"MD5 校验失败：{f['name']} 可能损坏，请重试")


def cmd_download(key, dest, file_filter=None, workers=8):
    if key not in DATASETS:
        print(f"[ERROR] 未知 key: {key}，可用: {', '.join(DATASETS)}")
        sys.exit(1)
    d = DATASETS[key]
    if d.get("hf"):
        print(f"[INFO] 下载 {d['name']}（HF 镜像）")
        print(f"[INFO] 来源: {d['url']}")
        download_hf(d["hf"], dest, max_workers=workers)
    elif d.get("direct"):
        print(f"[INFO] 下载 {d['name']}（直链）")
        print(f"[INFO] 来源: {d['url']}")
        download_direct(d["direct"], dest, file_filter)
    elif d["kaggle"]:
        if not check_kaggle_ready():
            sys.exit(1)
        os.makedirs(dest, exist_ok=True)
        print(f"[INFO] 下载 {d['name']}")
        print(f"[INFO] 来源: {d['url']}")
        cmd = ["kaggle", "datasets", "download", "-d", d["kaggle"],
               "-p", dest, "--unzip"]
        print(f"[INFO] 执行: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print(f"[ERROR] {key} 无自动下载源，请手动访问 {d['url']}")
        sys.exit(1)

    print("\n[DONE] 下载完成。目录结构预期：")
    print(f"  {d['layout']}")
    print(f"\n训练命令示例：")
    print(f"  python src/train.py --data-root <解压后的目录> --model efficientnet_b0 --epochs 20 --amp")
    if key in ("xray_pneumonia", "mri_brain"):
        print(f"\n[提示] 该数据集需要先划分 train/val：")
        print(f"  python scripts/split_dataset.py --src <解压后的train目录> --dst data/{key} "
              f"--ratios 0.7 0.15 0.15")


def main():
    p = argparse.ArgumentParser(description="公开医学影像数据集下载助手")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    dl = sub.add_parser("download")
    dl.add_argument("key", help="数据集 key，见 list 输出")
    dl.add_argument("--dest", default="data/raw", help="下载目标目录")
    dl.add_argument("--file", default=None,
                    help="只下载文件名包含该字符串的文件，如 1-766 或 README")
    dl.add_argument("--workers", type=int, default=8,
                    help="HF 下载并发数，被限流（429）时调小，如 4")
    args = p.parse_args()

    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "download":
        if getattr(args, "workers", 8) != 8:
            import functools
            global download_hf
            download_hf = functools.partial(download_hf, max_workers=args.workers)
        cmd_download(args.key, args.dest, args.file)


if __name__ == "__main__":
    main()
