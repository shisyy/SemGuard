import os
import re
import math

ROOT_DIR = os.environ.get(
    "SEMGUARD_RESULTS_ROOT",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        "hypernerf",
        "divid_msg48_800_continuous",
        "8frame",
    ),
)
OUTPUT_FILE = os.path.join(ROOT_DIR, "average_best_scores.txt")

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


def read_all_records(log_path):
    records = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            match = LINE_PATTERN.search(line)
            if match is None:
                continue

            iter_num = int(match.group(1))

            # 跳过 ITER 0，避免 psnr = inf
            if iter_num == 0:
                continue

            bit_acc = parse_value(match.group(2))
            psnr = parse_value(match.group(3))
            ssim = parse_value(match.group(4))
            lpips = parse_value(match.group(5))

            # 跳过 inf / nan，避免影响最大值判断
            if not math.isfinite(bit_acc):
                continue
            if not math.isfinite(psnr):
                continue

            records.append({
                "iter": iter_num,
                "Bit Acc": bit_acc,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "score": bit_acc + psnr,
                "raw_line": line,
            })

    return records


def find_best_record(records):
    """
    每个文件夹内部，选择 Bit Acc + psnr 最大的一行。
    如果相同，则优先 Bit Acc 更高、psnr 更高、iter 更小。
    """
    return max(
        records,
        key=lambda x: (
            x["score"],
            x["Bit Acc"],
            x["psnr"],
            -x["iter"],
        )
    )


def average(values):
    return sum(values) / len(values)


def main():
    best_records = {}
    skipped_folders = []

    for folder_name in sorted(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue

        log_path = os.path.join(folder_path, "train_scores.log")

        if not os.path.isfile(log_path):
            skipped_folders.append((folder_name, "没有 train_scores.log"))
            continue

        records = read_all_records(log_path)

        if len(records) == 0:
            skipped_folders.append((folder_name, "没有有效记录"))
            continue

        best_records[folder_name] = find_best_record(records)

    if len(best_records) == 0:
        raise RuntimeError("没有找到任何有效的 train_scores.log 记录。")

    folder_names = sorted(best_records.keys())

    avg_bit_acc = average([best_records[name]["Bit Acc"] for name in folder_names])
    avg_psnr = average([best_records[name]["psnr"] for name in folder_names])
    avg_ssim = average([best_records[name]["ssim"] for name in folder_names])
    avg_lpips = average([best_records[name]["lpips"] for name in folder_names])
    avg_score = average([best_records[name]["score"] for name in folder_names])

    sep = "————————————————————————————————————————"

    output_lines = []

    output_lines.append(sep)
    output_lines.append("每个文件夹中 Bit Acc + psnr 最大的原始行")
    output_lines.append(sep)
    output_lines.append("")

    for name in folder_names:
        record = best_records[name]
        output_lines.append(f"{name}")
        output_lines.append(record["raw_line"])
        output_lines.append(f"Bit Acc + psnr : {format_value(record['score'])}")
        output_lines.append("")

    output_lines.append(sep)
    output_lines.append("每个文件夹的最佳结果整理版")
    output_lines.append(sep)
    output_lines.append("")

    for name in folder_names:
        record = best_records[name]

        output_lines.append(
            f"{name} | "
            f"[ITER {record['iter']}] "
            f"Bit Acc : {format_value(record['Bit Acc'])} | "
            f"psnr : {format_value(record['psnr'])} | "
            f"ssim : {format_value(record['ssim'])} | "
            f"lpips : {format_value(record['lpips'])} | "
            f"Bit Acc + psnr : {format_value(record['score'])}"
        )

    output_lines.append("")
    output_lines.append(sep)
    output_lines.append("所有文件夹最佳结果的平均值")
    output_lines.append(sep)
    output_lines.append("")

    output_lines.append(
        f"Bit Acc : {format_value(avg_bit_acc)} | "
        f"psnr : {format_value(avg_psnr)} | "
        f"ssim : {format_value(avg_ssim)} | "
        f"lpips : {format_value(avg_lpips)} | "
        f"Bit Acc + psnr : {format_value(avg_score)}"
    )

    output_lines.append("")
    output_lines.append(f"参与平均的文件夹数量：{len(folder_names)}")

    if skipped_folders:
        output_lines.append("")
        output_lines.append(sep)
        output_lines.append("跳过的文件夹")
        output_lines.append(sep)
        output_lines.append("")

        for name, reason in skipped_folders:
            output_lines.append(f"{name}: {reason}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"完成，结果已保存到：{OUTPUT_FILE}")
    print(f"参与平均的文件夹数量：{len(folder_names)}")

    print("")
    print("平均结果：")
    print(
        f"Bit Acc : {format_value(avg_bit_acc)} | "
        f"psnr : {format_value(avg_psnr)} | "
        f"ssim : {format_value(avg_ssim)} | "
        f"lpips : {format_value(avg_lpips)} | "
        f"Bit Acc + psnr : {format_value(avg_score)}"
    )


if __name__ == "__main__":
    main()
