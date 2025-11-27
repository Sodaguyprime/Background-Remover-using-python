"""
Background Removal Script - Enhanced Version
- Faster processing with batch optimization
- Automatic output naming (filename_bg_removed.png)
- Multiple backend options: transparent-background, rembg (AI), or both
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List
import time

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_imports(backend='transparent'):
    """Check and provide installation instructions for missing packages"""
    missing = []
    instructions = []
    
    print(f"Checking dependencies for backend: {backend}...")
    
    # Common dependencies
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
    
    # Backend-specific
    if backend in ['transparent', 'both']:
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
    
    if backend in ['rembg', 'both']:
        try:
            import rembg
            print(f"✓ rembg installed (AI backend)")
        except ImportError:
            missing.append('rembg')
            instructions.append('pip install rembg[gpu]  # or rembg for CPU only')
    
    if missing:
        print("\n" + "=" * 70)
        print("MISSING DEPENDENCIES")
        print("=" * 70)
        print("\nInstall missing packages with:\n")
        for inst in instructions:
            print(f"  {inst}")
        
        if backend == 'rembg':
            print("\nFor AI backend (rembg):")
            print("  pip install rembg  # CPU version")
            print("  # OR")
            print("  pip install rembg[gpu]  # GPU version")
        elif backend == 'transparent':
            print("\nFor transparent-background:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            print("  pip install transparent-background")
        else:
            print("\nFor both backends:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            print("  pip install transparent-background rembg")
        
        print("\n" + "=" * 70)
        return False
    
    print("✓ All dependencies found!\n")
    return True


class BackgroundRemover:
    """Background removal with multiple backend options"""
    
    def __init__(self, backend='transparent', mode='base', use_gpu=False):
        """
        Initialize background remover
        
        Args:
            backend: 'transparent', 'rembg' (AI), or 'both'
            mode: For transparent backend - 'fast' or 'base'
            use_gpu: Whether to use GPU acceleration
        """
        self.backend = backend
        self.mode = mode
        self.remover_transparent = None
        self.remover_rembg = None
        
        # Setup device
        if backend in ['transparent', 'both']:
            import torch
            if use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
                print("Using GPU acceleration")
            else:
                self.device = 'cpu'
                print("Using CPU")
        
        # Initialize backends
        if backend in ['transparent', 'both']:
            self._init_transparent(mode)
        
        if backend in ['rembg', 'both']:
            self._init_rembg()
    
    def _init_transparent(self, mode):
        """Initialize transparent-background"""
        from transparent_background import Remover
        
        print(f"Initializing transparent-background ({mode} mode)...")
        print("(First run will download the model - may take a few minutes)")
        
        try:
            self.remover_transparent = Remover(mode=mode, jit=False, device=self.device)
            print("✓ Transparent-background model loaded!\n")
        except Exception as e:
            print(f"✗ Error loading transparent-background: {e}")
            raise
    
    def _init_rembg(self):
        """Initialize rembg (AI backend)"""
        from rembg import remove
        
        print("Initializing rembg (AI backend)...")
        print("(First run will download AI models - may take a few minutes)")
        
        try:
            # rembg will auto-download models on first use
            self.remover_rembg = remove
            print("✓ Rembg (AI) backend ready!\n")
        except Exception as e:
            print(f"✗ Error loading rembg: {e}")
            raise
    
    def process_image(self, input_path: str, output_path: str = None, 
                     threshold: int = 127) -> bool:
        """
        Process a single image
        
        Args:
            input_path: Input image path
            output_path: Output path (auto-generated if None)
            threshold: Threshold for alpha channel (0-255)
        """
        from PIL import Image
        import numpy as np
        
        try:
            # Check input exists
            if not os.path.exists(input_path):
                print(f"✗ Input file not found: {input_path}")
                return False
            
            # Auto-generate output path if not provided
            if output_path is None:
                input_pathobj = Path(input_path)
                output_path = str(input_pathobj.parent / f"{input_pathobj.stem}_bg_removed.png")
            
            print(f"Processing: {input_path}")
            
            # Load image
            try:
                img = Image.open(input_path)
            except Exception as e:
                print(f"✗ Error loading image: {e}")
                return False
            
            print(f"  Size: {img.width}x{img.height}")
            print(f"  Mode: {img.mode}")
            print(f"  Backend: {self.backend}")
            
            start_time = time.time()
            
            # Process based on backend
            if self.backend == 'rembg':
                output = self._process_rembg(img)
            elif self.backend == 'transparent':
                output = self._process_transparent(img)
            elif self.backend == 'both':
                # Try rembg first (usually more accurate), fallback to transparent
                try:
                    print("  Using rembg (AI)...")
                    output = self._process_rembg(img)
                except Exception as e:
                    print(f"  Rembg failed, trying transparent-background: {e}")
                    output = self._process_transparent(img)
            
            elapsed = time.time() - start_time
            print(f"  Processing time: {elapsed:.2f}s")
            
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
            output.save(output_path, 'PNG', optimize=True)
            
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
    
    def _process_transparent(self, img):
        """Process with transparent-background"""
        from PIL import Image
        
        # Convert to RGB if needed
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        print("  Removing background (transparent-background)...")
        return self.remover_transparent.process(img, type='rgba')
    
    def _process_rembg(self, img):
        """Process with rembg (AI)"""
        from PIL import Image
        import io
        
        print("  Removing background (rembg AI)...")
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Process
        output_bytes = self.remover_rembg(img_byte_arr.read())
        
        # Convert back to PIL Image
        return Image.open(io.BytesIO(output_bytes))
    
    def process_batch(self, input_dir: str, output_dir: str = None,
                     extensions=('.jpg', '.jpeg', '.png', '.webp', '.bmp'),
                     threshold: int = 127):
        """Process all images in a directory (optimized for speed)"""
        
        input_path = Path(input_dir)
        
        # Auto-generate output directory if not provided
        if output_dir is None:
            output_path = input_path / "bg_removed"
        else:
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
        
        print(f"Found {len(images)} images")
        print(f"Output directory: {output_path}\n")
        print("=" * 70)
        
        total_start = time.time()
        success = 0
        
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}]")
            output_file = output_path / f"{img_path.stem}_bg_removed.png"
            if self.process_image(str(img_path), str(output_file), threshold):
                success += 1
        
        total_elapsed = time.time() - total_start
        avg_time = total_elapsed / len(images) if images else 0
        
        print("=" * 70)
        print(f"Completed: {success}/{len(images)} images processed")
        print(f"Total time: {total_elapsed:.2f}s")
        print(f"Average time per image: {avg_time:.2f}s\n")


def main():
    # Check if files were dragged onto the exe
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Drag and drop mode
        print("\n" + "=" * 70)
        print("Background Removal Tool - Drag & Drop Mode")
        print("=" * 70 + "\n")
        
        # Check dependencies
        check_python_version()
        if not check_imports('both'):
            print("\nPress Enter to exit...")
            input()
            sys.exit(1)
        
        # Initialize with 'both' backend
        try:
            print("Initializing AI models (this may take a moment)...\n")
            remover = BackgroundRemover(backend='both', mode='base', use_gpu=False)
        except Exception as e:
            print(f"\nFailed to initialize: {e}")
            print("\nPress Enter to exit...")
            input()
            sys.exit(1)
        
        # Process all dragged files
        files = sys.argv[1:]
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff')
        valid_files = [f for f in files if f.lower().endswith(image_extensions)]
        
        if not valid_files:
            print("No valid image files found!")
            print("Supported formats: JPG, PNG, WEBP, BMP, GIF, TIFF")
            print("\nPress Enter to exit...")
            input()
            sys.exit(1)
        
        print(f"Processing {len(valid_files)} image(s)...\n")
        print("=" * 70)
        
        success = 0
        for i, file_path in enumerate(valid_files, 1):
            print(f"[{i}/{len(valid_files)}]")
            if remover.process_image(file_path, threshold=127):
                success += 1
        
        print("=" * 70)
        print(f"\n✓ Complete! {success}/{len(valid_files)} images processed")
        print("\nOutput files saved in the same folder as originals")
        print("(named: filename_bg_removed.png)\n")
        print("Press Enter to exit...")
        input()
        return
    
    # Command line mode
    parser = argparse.ArgumentParser(
        description='Enhanced Background Removal Tool - Fast & AI-Powered',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image (auto output naming)
  python script.py -i photo.jpg
  
  # Single image with custom output
  python script.py -i photo.jpg -o custom_name.png
  
  # Batch process folder (creates ./photos/bg_removed/)
  python script.py -i ./photos -b
  
  # Use AI backend (rembg) for better accuracy
  python script.py -i photo.jpg --backend rembg
  
  # Fast mode with transparent-background
  python script.py -i photo.jpg -m fast
  
  # Try both backends (rembg first, fallback to transparent)
  python script.py -i photo.jpg --backend both
  
  # Drag and Drop:
  Simply drag images onto the .exe file to process them automatically!

Installation:
  # For transparent-background (default, fast)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install transparent-background pillow numpy
  
  # For rembg AI backend (more accurate)
  pip install rembg pillow numpy
  
  # For both
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install transparent-background rembg pillow numpy

Backends:
  - transparent: Fast, good quality (default)
  - rembg: AI-powered, most accurate, slower
  - both: Try rembg first, fallback to transparent
  
Modes (transparent backend only):
  - base: High quality (default, ~170MB model)
  - fast: Fast processing (~39MB model)
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input image or directory')
    parser.add_argument('-o', '--output', default=None,
                       help='Output image or directory (auto-generated if not specified)')
    parser.add_argument('-b', '--batch', action='store_true',
                       help='Batch process directory')
    parser.add_argument('--backend', default='transparent',
                       choices=['transparent', 'rembg', 'both'],
                       help='Backend to use (transparent=fast, rembg=AI/accurate, both=try both)')
    parser.add_argument('-m', '--mode', default='base',
                       choices=['base', 'fast'],
                       help='Mode for transparent backend (base=quality, fast=speed)')
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
    print("Enhanced Background Removal Tool")
    print("=" * 70 + "\n")
    
    # Check dependencies
    check_python_version()
    if not check_imports(args.backend):
        sys.exit(1)
    
    # Initialize
    try:
        remover = BackgroundRemover(
            backend=args.backend,
            mode=args.mode,
            use_gpu=args.gpu
        )
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
