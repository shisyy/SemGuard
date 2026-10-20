import os
import re
import math

ROOT_DIR = os.environ.get(
    "SEMGUARD_RESULTS_ROOT",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        "dnerf",
        "divid_msg48_800_continuous",
        "8frame",
    ),
)
TARGET_ITER = 100
OUTPUT_FILE = os.path.join(ROOT_DIR, "average_epoch100_scores.txt")

LINE_PATTERN = re.compile(
    r"\[ITER\s+(\d+)\]\s*"
    r"Bit Acc\s*:\s*([^\s|]+)\s*\|\s*"
    r"psnr\s*:\s*([^\s|]+)\s*\|\s*"
    r"ssim\s*:\s*([^\s|]+)\s*\|\s*"
    r"lpips\s*:\s*([^\s|]+)"
)


def parse_value(value):
    value = value.strip().lower()

    if value == "inf":
        return math.inf
    elif value == "-inf":
        return -math.inf
    else:
        return float(value)


def format_value(value):
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.4f}"


def read_iter_records(log_path):
    records = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = LINE_PATTERN.search(line)
            if match is None:
                continue

            iter_num = int(match.group(1))
            if iter_num != TARGET_ITER:
                continue

            record = {
                "Bit Acc": parse_value(match.group(2)),
                "psnr": parse_value(match.group(3)),
                "ssim": parse_value(match.group(4)),
                "lpips": parse_value(match.group(5)),
            }

            records.append(record)

    return records


def average(values):
    return sum(values) / len(values)


def main():
    all_records = {}
    skipped_folders = []

    for folder_name in sorted(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue

        log_path = os.path.join(folder_path, "train_scores.log")

        if not os.path.isfile(log_path):
            skipped_folders.append(folder_name)
            continue

        records = read_iter_records(log_path)

        if len(records) == 0:
            skipped_folders.append(folder_name)
            continue

        all_records[folder_name] = records

    if len(all_records) == 0:
        raise RuntimeError(f"没有找到任何包含 [ITER {TARGET_ITER}] 的 train_scores.log")

    folder_names = sorted(all_records.keys())

    record_counts = [len(all_records[name]) for name in folder_names]

    if len(set(record_counts)) != 1:
        print("警告：不同文件夹中的 [ITER 100] 行数不一致。")
        for name in folder_names:
            print(f"{name}: {len(all_records[name])} lines")

        min_count = min(record_counts)
        print(f"只会对前 {min_count} 条 [ITER 100] 记录按顺序求平均。")
    else:
        min_count = record_counts[0]

    output_lines = []

    for record_idx in range(min_count):
        avg_bit_acc = average([
            all_records[name][record_idx]["Bit Acc"]
            for name in folder_names
        ])

        avg_psnr = average([
            all_records[name][record_idx]["psnr"]
            for name in folder_names
        ])

        avg_ssim = average([
            all_records[name][record_idx]["ssim"]
            for name in folder_names
        ])

        avg_lpips = average([
            all_records[name][record_idx]["lpips"]
            for name in folder_names
        ])

        line = (
            f"[ITER {TARGET_ITER}] "
            f"Bit Acc : {format_value(avg_bit_acc)} | "
            f"psnr : {format_value(avg_psnr)} | "
            f"ssim : {format_value(avg_ssim)} | "
            f"lpips : {format_value(avg_lpips)}"
        )

        output_lines.append(line)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"完成，结果已保存到：{OUTPUT_FILE}")
    print(f"参与平均的文件夹数量：{len(folder_names)}")
    print("参与平均的文件夹：")
    for name in folder_names:
        print(f"  - {name}")

    if skipped_folders:
        print("跳过的文件夹：")
        for name in skipped_folders:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
