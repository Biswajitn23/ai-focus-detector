"""
Automatically download focus detection training images from Bing Images.
"""

import os
import sys
import shutil

try:
    from bing_image_downloader import downloader
except ImportError:
    print("Installing bing-image-downloader...")
    os.system("pip install bing-image-downloader")
    from bing_image_downloader import downloader

def download_images(query, output_dir, num_images=50, limit_per_query=25):
    """Download images from Bing for a given query"""
    print(f"Downloading {num_images} images for: '{query}'")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        downloader.download(
            query,
            limit=min(num_images, limit_per_query),
            output_dir="dataset",
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )
        
        # Move downloaded images to target directory
        source_dir = f"dataset/{query}"
        if os.path.exists(source_dir):
            files = os.listdir(source_dir)
            count = 0
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        src = os.path.join(source_dir, file)
                        dst = os.path.join(output_dir, file)
                        shutil.copy2(src, dst)
                        count += 1
                    except Exception as e:
                        print(f"  Error copying {file}: {e}")
            
            print(f"✓ Downloaded {count} images to {output_dir}")
            return count
    except Exception as e:
        print(f"✗ Error downloading images for '{query}': {e}")
        return 0

def main():
    print("Focus Detection Auto-Downloader")
    print("="*60)
    
    os.makedirs("data/focused", exist_ok=True)
    os.makedirs("data/not_focused", exist_ok=True)
    
    # Download focused images
    print("\n1. Downloading FOCUSED images...")
    focused_queries = [
        "person looking at computer screen",
        "person looking at camera",
        "student concentrating on work"
    ]
    
    focused_count = 0
    for query in focused_queries:
        count = download_images(query, "data/focused", num_images=20)
        focused_count += count
    
    # Download not focused images
    print("\n2. Downloading NOT FOCUSED images...")
    not_focused_queries = [
        "person looking away from screen",
        "person distracted sleepy",
        "person turned away",
        "person looking sideways"
    ]
    
    not_focused_count = 0
    for query in not_focused_queries:
        count = download_images(query, "data/not_focused", num_images=20)
        not_focused_count += count
    
    # Cleanup
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")
    
    print(f"\n{'='*60}")
    print("Download Complete!")
    print(f"{'='*60}")
    print(f"Focused images: {focused_count}")
    print(f"Not Focused images: {not_focused_count}")
    print(f"Total: {focused_count + not_focused_count}")
    
    if focused_count + not_focused_count > 10:
        print("\n✓ Dataset ready! Training with:")
        print("python train_focus.py --data_dir data --output focus_model.pkl")
    else:
        print("\n⚠ Limited images downloaded. Try running again or add more manually.")

if __name__ == "__main__":
    main()
