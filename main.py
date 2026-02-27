import os
import argparse
from src.processor import BatchProcessor
from src.drive_manager import DriveManager

def main():
    parser = argparse.ArgumentParser(description="Bulk Watermark Removal Tool")
    parser.add_argument("--input", type=str, default="data/input", help="Local input folder or Drive folder ID")
    parser.add_argument("--output", type=str, default="data/output", help="Local output folder")
    parser.add_argument("--drive", action="store_true", help="Use Google Drive for input")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLOv11 weights")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()

    # Initialize processor
    processor = BatchProcessor(model_path=args.model, num_workers=args.workers)
    
    # Ensure required directories exist
    os.makedirs("data/masks", exist_ok=True)

    image_paths = []

    if args.drive:
        print("Connecting to Google Drive...")
        drive = DriveManager()
        
        # 1. Parse parent folder ID from link
        parent_id = drive.extract_id_from_link(args.input)
        print(f"Parent Folder ID: {parent_id}")

        # 2. Find 'image' subfolder
        input_folder_id = drive.get_subfolder_id(parent_id, "image")
        if not input_folder_id:
            print("Error: Could not find a folder named 'image' in the provided Drive link.")
            return
        
        # 3. List and download files
        files = drive.list_files(input_folder_id)
        if not files:
            print("No images found in the 'image' folder on Drive.")
            return
        print(f"Found {len(files)} images in 'image' folder.")
        
        temp_input_dir = "data/input_temp"
        temp_output_dir = "data/output_temp"
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)
        
        downloaded_paths = []
        for f in files:
            path = drive.download_file(f['id'], os.path.join(temp_input_dir, f['name']))
            downloaded_paths.append(path)
        
        # 4. Process locally
        print(f"Starting processing of {len(downloaded_paths)} images...")
        processor.process_batch(downloaded_paths, temp_output_dir)
        
        # 5. Find or create 'output image' folder on Drive
        output_folder_id = drive.get_subfolder_id(parent_id, "output image")
        if not output_folder_id:
            print("Creating 'output image' folder on Drive...")
            output_folder_id = drive.create_folder(parent_id, "output image")
            
        # 6. Upload processed images
        print("Uploading processed images to Drive...")
        processed_files = os.listdir(temp_output_dir)
        for filename in processed_files:
            file_path = os.path.join(temp_output_dir, filename)
            if os.path.isfile(file_path):
                drive.upload_file(file_path, output_folder_id)
                print(f"Uploaded: {filename}")
        
        print("\nDrive sync complete!")
        
    else:
        # Local processing
        if not os.path.exists(args.input):
            print(f"Input path {args.input} does not exist.")
            return
            
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

        print(f"Starting processing of {len(image_paths)} images...")
        processor.process_batch(image_paths, args.output)

if __name__ == "__main__":
    main()
