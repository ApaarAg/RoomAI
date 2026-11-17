import os
import zipfile
from pathlib import Path

WEB_DIR = Path("web")
OUTPUT_ZIP = Path("roomai-frontend.zip")

EXCLUDE_DIRS = {
    "node_modules",
    ".next",
    "__pycache__",
    ".git"
}

EXCLUDE_FILES = {
    "tsconfig.tsbuildinfo",
    ".DS_Store"
}

def create_zip():
    print(f"Creating {OUTPUT_ZIP} from {WEB_DIR}...")
    
    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(WEB_DIR):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                if file in EXCLUDE_FILES or file.endswith(".pyc"):
                    continue
                
                full_path = Path(root) / file
                rel_path = full_path.relative_to(WEB_DIR)
                arcname = Path("roomai-frontend") / rel_path
                
                file_size = full_path.stat().st_size
                total_files += 1
                total_uncompressed_bytes += file_size
                
                zipf.write(full_path, arcname=str(arcname))
                print(f"  + {arcname} ({file_size:,} bytes)")

    zip_size = OUTPUT_ZIP.stat().st_size
    print("\n" + "="*60)
    print(f"SUCCESSFULLY CREATED: {OUTPUT_ZIP.resolve()}")
    print(f"Total files packaged: {total_files}")
    print(f"Uncompressed size:   {total_uncompressed_bytes / (1024*1024):.2f} MB")
    print(f"Compressed zip size: {zip_size / (1024*1024):.2f} MB")
    print("="*60)

if __name__ == "__main__":
    create_zip()
