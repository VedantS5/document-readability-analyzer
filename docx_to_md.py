import subprocess
import sys
import os

# Function to check if a package is installed
def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# Function to install package if not installed
def install_package(package_name):
    print(f"{package_name} not found, installing...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f"{package_name} installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}. Please install it manually.")
        return False

# Ensure docx2python is installed
if not is_package_installed('docx2python'):
    success = install_package('docx2python')
    if not success:
        sys.exit(1)

# Import docx2python after ensuring it's installed
from docx2python import docx2python

# Function to ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Function to convert docx files to markdown text files
def convert_docx_to_md(input_dir, output_dir):
    """
    Convert all .docx files in input_dir to markdown files in output_dir.
    
    Args:
        input_dir (str): Directory containing .docx files
        output_dir (str): Directory to save converted .md files
    """
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Count files for reporting
    total_files = 0
    successful_files = 0
    failed_files = 0
    
    # Process each .docx file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.docx'):
            total_files += 1
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.md'
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                print(f"Processing: {filename}")
                with docx2python(input_path, html=True) as docx_content:
                    text = docx_content.text
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"Success: {filename} -> {output_filename}")
                successful_files += 1
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                failed_files += 1
    
    # Report summary
    if total_files == 0:
        print(f"No .docx files found in '{input_dir}'.")
    else:
        print(f"\nConversion Summary:")
        print(f"Total .docx files: {total_files}")
        print(f"Successfully converted: {successful_files}")
        print(f"Failed to convert: {failed_files}")

# Main function
def main():
    """Main function to parse command-line arguments and run the conversion."""
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    convert_docx_to_md(input_dir, output_dir)

# Run the script if it's the main module
if __name__ == '__main__':
    main()

