import re
from pathlib import Path
import argparse

def extract_h_lines(input_file):
    text = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f.read().splitlines():
            if line.startswith('H'):
                text.append(line)
    return text

def main(path):
    # Assume that the file name does not change
    test_file = f"{path}/generate-test.txt"
    out_file =f"{path}/generate-test.hyp"

    text = extract_h_lines(test_file)
    sorted_text = sorted(text, key=lambda x: int(x.split("\t")[0][2:]))
    with open(out_file, 'w', encoding='utf-8') as f:
        for t in sorted_text:
            f.write(t.split("\t")[-1] + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process result directory.')
    parser.add_argument('--path', type=str, help='Path to the result directory', required=True)

    args = parser.parse_args()
    main(args.path)
