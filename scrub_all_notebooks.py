import json
from pathlib import Path
import argparse

def load_patterns(pattern_file: Path) -> dict:
    if not pattern_file.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_file}")
    with open(pattern_file, "r", encoding="utf-8") as f:
        return json.load(f)

def scrub_notebook(file_path: Path, patterns: dict):
    try:
        with file_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)

        changed = False

        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                for i, line in enumerate(cell.get("source", [])):
                    for target, replacement in patterns.items():
                        if target in line:
                            cell["source"][i] = line.replace(target, replacement)
                            changed = True
                for output in cell.get("outputs", []):
                    if "text" in output:
                        for i, line in enumerate(output["text"]):
                            for target, replacement in patterns.items():
                                if target in line:
                                    output["text"][i] = line.replace(target, replacement)
                                    changed = True

        if changed:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(nb, f, indent=2)
            print(f"Scrubbed: {file_path}")

    except Exception as e:
        print(f"Failed to scrub {file_path}: {e}")

def scrub_all_notebooks(root_dir: Path, patterns: dict):
    for file_path in root_dir.rglob("*.ipynb"):
        scrub_notebook(file_path, patterns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively scrub internal terms from Jupyter notebooks.")
    parser.add_argument(
        "--target-dir",
        type=str,
        default="./",
        help="Target directory to search recursively (default: ./default)"
    )
    parser.add_argument(
        "--pattern-file",
        type=str,
        default="scrub_patterns.json",
        help="Path to JSON file with pattern replacements (default: scrub_patterns.json)"
    )

    args = parser.parse_args()
    target_dir = Path(args.target_dir)
    pattern_file = Path(args.pattern_file)

    try:
        patterns = load_patterns(pattern_file)
        scrub_all_notebooks(target_dir, patterns)
    except Exception as e:
        print(f"Error: {e}")
