import os
import argparse
from src.processor import BatchProcessor
from src.drive_manager import DriveManager

def main():
    parser = argparse.ArgumentParser(description="Bulk Watermark Removal Tool")
    parser.add_argument("--input", type=str, default="data/input", help="Local input folder or Drive parent folder ID/URL")
    parser.add_argument("--output", type=str, default="data/output", help="Local output folder")
    parser.add_argument("--drive", action="store_true", help="Use Google Drive for input/output")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLOv11 weights")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    # Initialize processor
    processor = BatchProcessor(model_path=args.model, num_workers=args.workers)


    if args.drive:
        print("Connecting to Google Drive...")
        drive = DriveManager()

        # 1. Parse parent folder ID from link
        parent_id = drive.extract_id_from_link(args.input)
        print(f"Parent Folder ID: {parent_id}")

        # 2. Find 'image' subfolder (input)
        input_folder_id = drive.get_subfolder_id(parent_id, "image")
        if not input_folder_id:
            print("Error: Could not find a folder named 'image' in the provided Drive link.")
            return

        # 3. List and download files from 'image' folder
        files = drive.list_files(input_folder_id)
        if not files:
            print("No images found in the 'image' folder on Drive.")
            return
        print(f"Found {len(files)} images in 'image' folder.")

        # 4. Prepare local temp directories
        temp_input_dir = "data/input_temp"
        temp_clean_dir = "data/clean_temp"
        temp_failed_dir = "data/failed_temp"
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_clean_dir, exist_ok=True)
        os.makedirs(temp_failed_dir, exist_ok=True)

        # 5. Download all images locally
        print("Downloading images from Drive...")
        downloaded = []
        for f in files:
            local_path = os.path.join(temp_input_dir, f['name'])
            drive.download_file(f['id'], local_path)
            downloaded.append(local_path)
        print(f"Downloaded {len(downloaded)} images.")

        # 6. Process images — returns lists of (success_paths, failed_paths)
        print(f"Starting processing of {len(downloaded)} images...")
        clean_paths, failed_paths = processor.process_batch_with_results(
            downloaded, temp_clean_dir, temp_failed_dir
        )

        # 7. Find or create 'clean image' folder on Drive
        clean_folder_id = drive.get_subfolder_id(parent_id, "clean image")
        if not clean_folder_id:
            print("Creating 'clean image' folder on Drive...")
            clean_folder_id = drive.create_folder(parent_id, "clean image")

        # 8. Find or create 'failed image' folder on Drive
        failed_folder_id = drive.get_subfolder_id(parent_id, "failed image")
        if not failed_folder_id:
            print("Creating 'failed image' folder on Drive...")
            failed_folder_id = drive.create_folder(parent_id, "failed image")

        # 9. Upload cleaned images
        print(f"\nUploading {len(clean_paths)} clean images to 'clean image' folder...")
        for file_path in clean_paths:
            drive.upload_file(file_path, clean_folder_id)
            print(f"  ✔ Uploaded: {os.path.basename(file_path)}")

        # 10. Upload failed images
        if failed_paths:
            print(f"\nUploading {len(failed_paths)} failed images to 'failed image' folder...")
            for file_path in failed_paths:
                drive.upload_file(file_path, failed_folder_id)
                print(f"  ✘ Failed upload: {os.path.basename(file_path)}")
        else:
            print("\nNo failed images — all processed successfully!")

        print("\n✅ Drive sync complete!")
        print(f"   Clean images  → 'clean image'  ({len(clean_paths)} files)")
        print(f"   Failed images → 'failed image' ({len(failed_paths)} files)")

    else:
        # ── Local processing ──────────────────────────────────────────────
        if not os.path.exists(args.input):
            print(f"Input path {args.input} does not exist.")
            return

        image_paths = []
        if os.path.isfile(args.input):
            if args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_paths.append(args.input)
        else:
            for f in os.listdir(args.input):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_paths.append(os.path.join(args.input, f))

        if not image_paths:
            print("No images found to process.")
            return

        failed_dir = os.path.join(args.output, "failed")
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(failed_dir, exist_ok=True)

        print(f"Starting processing of {len(image_paths)} images...")
        clean_paths, failed_paths = processor.process_batch_with_results(
            image_paths, args.output, failed_dir
        )
        print(f"\n✅ Done — {len(clean_paths)} clean, {len(failed_paths)} failed.")

if __name__ == "__main__":
    main()
