import os
import argparse
from src.processor import BatchProcessor

def main():
    parser = argparse.ArgumentParser(description="Bulk Watermark Removal Tool")
    parser.add_argument("--input", type=str, default="data/input", help="Local input folder path")
    parser.add_argument("--output", type=str, default="data/output", help="Local output folder path")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLOv11 weights")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--watch", action="store_true", help="Watch input folder for new images continuously")
    parser.add_argument("--interval", type=int, default=2, help="Check interval in seconds for watch mode")
    parser.add_argument("--force", action="store_true", help="Process all images even if they were processed before")

    args = parser.parse_args()

    # Initialize processor
    processor = BatchProcessor(model_path=args.model, num_workers=args.workers, force=args.force)

    print(f"🚀 Watermark Removal Tool Started")
    if args.watch:
        print(f"👀 Watching folder: {args.input}")
        print(f"📁 Output folder: {args.output}")
        print("💡 Drop images into the input folder to process them automatically (Ctrl+C to stop).")
    
    try:
        while True:
            # ── Local processing ──────────────────────────────────────────────
            if not os.path.exists(args.input):
                print(f"Input path {args.input} does not exist.")
                if not args.watch: return
                time.sleep(args.interval)
                continue

            image_paths = []
            if os.path.isfile(args.input):
                if args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_paths.append(args.input)
            else:
                for f in os.listdir(args.input):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        # Only add if not already in processed_files to avoid re-scanning everything every loop
                        if f not in processor.processed_files:
                            image_paths.append(os.path.join(args.input, f))

            if image_paths:
                failed_dir = os.path.join(args.output, "failed")
                os.makedirs(args.output, exist_ok=True)
                os.makedirs(failed_dir, exist_ok=True)

                print(f"\n✨ Found {len(image_paths)} new image(s). Processing...")
                clean_paths, failed_paths = processor.process_batch_with_results(
                    image_paths, args.output, failed_dir
                )
                print(f"✅ Batch complete — {len(clean_paths)} clean, {len(failed_paths)} failed.")
            
            if not args.watch:
                break
            
            import time
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n👋 Stopping watcher...")

if __name__ == "__main__":
    main()

