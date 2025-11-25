"""
Background Removal Script - Compatible with Python 3.13
Uses transparent-background library which supports newer Python versions
"""

import os
import sys
from pathlib import Path
import argparse

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_imports():
    """Check and provide installation instructions for missing packages"""
    missing = []
    instructions = []
    
    print("Checking dependencies...")
    
    try:
        from PIL import Image
        print(f"✓ Pillow installed")
    except ImportError:
        missing.append('pillow')
        instructions.append('pip install pillow')
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        missing.append('numpy')
        instructions.append('pip install numpy')
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        missing.append('torch')
        instructions.append('pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu')
    
    try:
        import transparent_background
        print(f"✓ transparent-background installed")
    except ImportError:
        missing.append('transparent-background')
        instructions.append('pip install transparent-background')
    
    if missing:
        print("\n" + "=" * 70)
        print("MISSING DEPENDENCIES")
        print("=" * 70)
        print("\nInstall missing packages with:\n")
        for inst in instructions:
            print(f"  {inst}")
        print("\nOr install all at once:")
        print("\n  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install transparent-background pillow numpy")
        print("\n" + "=" * 70)
        return False
    
    print("✓ All dependencies found!\n")
    return True

check_python_version()

if not check_imports():
    sys.exit(1)

# Now import everything
from PIL import Image
import numpy as np
import torch
from transparent_background import Remover


class BackgroundRemover:
    """Background removal using transparent-background library"""
    
    def __init__(self, mode='base', use_gpu=False):
        """
        Initialize background remover
        
        Args:
            mode: 'fast' or 'base' (base is higher quality)
            use_gpu: Whether to use GPU acceleration
        """
        self.mode = mode
        
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            print("Using GPU acceleration")
        else:
            self.device = 'cpu'
            print("Using CPU")
        
        print(f"Initializing {mode} model...")
        print("(First run will download the model - may take a few minutes)")
        
        try:
            self.remover = Remover(mode=mode, jit=False, device=self.device)
            print("✓ Model loaded successfully!\n")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. Check internet connection (needed for first download)")
            print("2. Try reinstalling: pip uninstall transparent-background && pip install transparent-background")
            print("3. Try fast mode: -m fast")
            raise
    
    def process_image(self, input_path: str, output_path: str, 
                     threshold: int = 127) -> bool:
        """
        Process a single image
        
        Args:
            input_path: Input image path
            output_path: Output image path
            threshold: Threshold for alpha channel (0-255, lower = more transparent)
        """
        try:
            # Check input exists
            if not os.path.exists(input_path):
                print(f"✗ Input file not found: {input_path}")
                return False
            
            print(f"Processing: {input_path}")
            
            # Load image
            try:
                img = Image.open(input_path)
            except Exception as e:
                print(f"✗ Error loading image: {e}")
                return False
            
            # Convert to RGB if needed
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            print(f"  Size: {img.width}x{img.height}")
            print(f"  Mode: {img.mode}")
            print("  Removing background...")
            
            # Process
            try:
                output = self.remover.process(img, type='rgba')
            except Exception as e:
                print(f"✗ Error during processing: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Apply threshold if needed
            if threshold != 127:
                print(f"  Applying threshold: {threshold}")
                output_array = np.array(output)
                alpha = output_array[:, :, 3]
                alpha[alpha < threshold] = 0
                output_array[:, :, 3] = alpha
                output = Image.fromarray(output_array)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save
            print("  Saving...")
            try:
                if output_path.lower().endswith('.png'):
                    output.save(output_path, 'PNG', optimize=True)
                elif output_path.lower().endswith(('.jpg', '.jpeg')):
                    # Convert to RGB with white background for JPEG
                    bg = Image.new('RGB', output.size, (255, 255, 255))
                    bg.paste(output, mask=output.split()[3])
                    bg.save(output_path, 'JPEG', quality=95, optimize=True)
                else:
                    if not output_path.endswith('.png'):
                        output_path += '.png'
                    output.save(output_path, 'PNG', optimize=True)
            except Exception as e:
                print(f"✗ Error saving: {e}")
                return False
            
            # Verify
            if os.path.exists(output_path):
                size_kb = os.path.getsize(output_path) / 1024
                print(f"✓ Saved: {output_path} ({size_kb:.1f} KB)\n")
                return True
            else:
                print(f"✗ Output not created\n")
                return False
            
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_batch(self, input_dir: str, output_dir: str,
                     extensions=('.jpg', '.jpeg', '.png', '.webp', '.bmp'),
                     threshold: int = 127):
        """Process all images in a directory"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find images
        images = []
        for ext in extensions:
            images.extend(input_path.glob(f'*{ext}'))
            images.extend(input_path.glob(f'*{ext.upper()}'))
        
        images = list(set(images))
        
        if not images:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(images)} images\n")
        print("=" * 70)
        
        success = 0
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}]")
            output_file = output_path / f"{img_path.stem}_no_bg.png"
            if self.process_image(str(img_path), str(output_file), threshold):
                success += 1
        
        print("=" * 70)
        print(f"Completed: {success}/{len(images)} images processed\n")


def main():
    parser = argparse.ArgumentParser(
        description='Background Removal Tool - Python 3.13 Compatible',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python script.py -i photo.jpg -o output.png
  
  # Batch process folder
  python script.py -i ./photos -o ./output -b
  
  # Fast mode (lower quality, faster)
  python script.py -i photo.jpg -o output.png -m fast
  
  # Adjust transparency threshold (0-255)
  python script.py -i photo.jpg -o output.png -t 100

Installation:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install transparent-background pillow numpy

Modes:
  - base: High quality (default, slower, ~170MB model)
  - fast: Fast processing (lower quality, ~39MB model)
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input image or directory')
    parser.add_argument('-o', '--output', required=True,
                       help='Output image or directory')
    parser.add_argument('-b', '--batch', action='store_true',
                       help='Batch process directory')
    parser.add_argument('-m', '--mode', default='base',
                       choices=['base', 'fast'],
                       help='Processing mode (base=quality, fast=speed)')
    parser.add_argument('-t', '--threshold', type=int, default=127,
                       help='Alpha threshold 0-255 (lower=more transparent)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0 <= args.threshold <= 255:
        print("Error: threshold must be between 0 and 255")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Background Removal Tool")
    print("=" * 70 + "\n")
    
    # Initialize
    try:
        remover = BackgroundRemover(mode=args.mode, use_gpu=args.gpu)
    except Exception as e:
        print(f"\nFailed to initialize: {e}")
        sys.exit(1)
    
    # Process
    if args.batch:
        remover.process_batch(args.input, args.output, threshold=args.threshold)
    else:
        remover.process_image(args.input, args.output, threshold=args.threshold)


if __name__ == '__main__':
    main()
