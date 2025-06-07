#!/usr/bin/env python3

import subprocess
import sys
import os
import platform

def run_command(cmd, description=""):
    """Run a command and handle errors gracefully."""
    if description:
        print(f"🔧 {description}...")
    
    try:
        # Split command into list to avoid shell interpretation issues
        import shlex
        cmd_list = shlex.split(cmd)
        result = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def check_conda():
    """Check if conda is available."""
    try:
        subprocess.run(["conda", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_virtual_env():
    """Check if we're in a virtual environment (including conda)."""
    # Check for conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env and conda_env != 'base':
        return True, f"conda:{conda_env}"
    
    # Check for traditional virtual environments
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = os.environ.get('VIRTUAL_ENV', 'unknown')
        return True, f"venv:{os.path.basename(venv_path)}"
    
    return False, None

def main():
    print("🦄 UWUifier Installation Script")
    print("===============================")
    
    # Check environment
    has_conda = check_conda()
    in_venv, venv_info = check_virtual_env()
    is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
    
    print(f"✅ Python version: {sys.version}")
    print(f"{'✅' if has_conda else '❌'} Conda available: {has_conda}")
    print(f"{'✅' if in_venv else '⚠️ '} Virtual environment: {venv_info if in_venv else 'None'}")
    print(f"{'✅' if is_apple_silicon else 'ℹ️ '} Apple Silicon: {is_apple_silicon}")
    
    if not in_venv:
        print("\n⚠️  Consider using a virtual environment:")
        if has_conda:
            print("conda create -n uwuifier python=3.11")
            print("conda activate uwuifier")
        else:
            print("python -m venv uwuifier")
            if platform.system() == "Windows":
                print("uwuifier\\Scripts\\activate")
            else:
                print("source uwuifier/bin/activate")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("\n🚀 Starting installation...")
    
    # Install PyTorch first
    if has_conda and is_apple_silicon:
        success = run_command(
            "conda install pytorch torchvision -c pytorch -y",
            "Installing PyTorch with conda (Apple Silicon optimized)"
        )
    else:
        success = run_command(
            "pip install torch>=1.9.0 torchvision>=0.10.0",
            "Installing PyTorch"
        )
    
    if not success:
        print("❌ PyTorch installation failed. This is critical.")
        sys.exit(1)
    
    # Install dependencies in order
    dependencies = [
        ("pip install Pillow>=8.0.0 numpy>=1.21.0", "Installing core dependencies"),
        ("pip install diffusers>=0.21.0 transformers>=4.21.0 huggingface_hub>=0.16.0 safetensors>=0.3.0", "Installing Hugging Face ecosystem"),
        ("pip install ultralytics>=8.0.0 controlnet_aux>=0.0.6", "Installing computer vision dependencies"),
        ("pip install onnxruntime>=1.15.0", "Installing ONNX Runtime"),
        ("pip install accelerate>=0.20.0", "Installing acceleration dependencies"),
    ]
    
    for cmd, desc in dependencies:
        if not run_command(cmd, desc):
            print(f"⚠️  Failed to install: {desc}")
    
    # Handle xformers specially for Apple Silicon
    if is_apple_silicon:
        print("🍎 Apple Silicon detected - trying alternative xformers installation...")
        
        # Try pre-built wheel first
        xformers_success = run_command(
            "pip install xformers --no-build-isolation",
            "Installing xformers (no build isolation)"
        )
        
        if not xformers_success:
            print("⚠️  Trying alternative xformers installation methods...")
            
            # Try with specific version that might have pre-built wheels
            xformers_success = run_command(
                "pip install xformers==0.0.22.post7 --no-deps",
                "Installing specific xformers version"
            )
            
            if not xformers_success:
                print("⚠️  xformers installation failed - this is optional")
                print("📝 Note: On Apple Silicon, xformers often fails to compile.")
                print("   Your UWUifier will still work without it, just potentially slower.")
    else:
        # Standard installation for other platforms
        print("🔄 Installing xformers (this might take a while and is optional)...")
        if not run_command("pip install xformers>=0.0.20", "Installing xformers"):
            print("⚠️  xformers installation failed - this is optional and you can continue without it")
    
    print("\n" + "="*50)
    print("✅ Installation complete!")
    print("="*50)
    
    # Test critical imports
    print("\n🧪 Testing critical imports...")
    test_imports = [
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
    ]
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - CRITICAL ERROR")
    
    # Test optional imports
    optional_imports = [
        ("xformers", "xformers (optional)"),
    ]
    
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - not available (optional)")
    
    print("\n🎉 You can now run:")
    print("python main.py")
    
    if is_apple_silicon:
        print("\n🍎 Apple Silicon tips:")
        print("• Your GPU acceleration will use Metal Performance Shaders")
        print("• If you encounter memory issues, try smaller image sizes")
        print("• Some operations may be slower without xformers, but everything should work")

if __name__ == "__main__":
    main()