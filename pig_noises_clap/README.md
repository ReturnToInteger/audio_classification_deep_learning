# Audio Processing Script

This Python script uses machine learning to automatically classify and segment audio files based on predefined labels. It's specifically designed for analyzing pig sounds but can be adapted for other audio classification tasks.

## What it does

The script processes audio files by:
1. **Splitting audio into frames** - Divides long audio files into smaller overlapping segments
2. **Classifying each frame** - Uses a CLAP (Contrastive Language-Audio Pre-training) model to match audio segments with text labels
3. **Creating labeled segments** - Groups similar frames together and exports them as separate audio files
4. **Generating Audacity labels** - Creates label files that can be imported into Audacity for manual review

## Default Configuration

The script comes with predefined settings for pig sound analysis:
- **Labels**: Background noise, brief peak events, pig sneeze, pig oink
- **Frame size**: 1 second with 0.5 second overlap
- **Sample rate**: 16,000 Hz
- **Batch processing**: 70 frames at a time

## Environment Setup

This project uses a mamba/conda environment for dependency management. This ensures reproducible results and easier setup.

### Install using the provided environment file (recommended)
```bash
# Create environment from environment.yml file
mamba env create -f environment.yml

# Activate the environment
mamba activate audioenv

# Install CUDA version of PyTorch (optional, for GPU acceleration)
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Note: The environment.yml includes CPU-only PyTorch. For GPU acceleration, run the optional CUDA installation command above.


## Usage

### Basic usage (with defaults):
```bash
python audio_processor.py
```

### Custom configuration:
```bash
python audio_processor.py --labels "dog bark" "cat meow" "silence" --source "path/to/audio/files" --frame_s 2.0 --hop_s 1.0
```

### Evaluating Results

After processing, you can visualize the detection results over time:

```bash
python eval_script.py --folder "path/to/output/folder"
```

This evaluation script:
- Reads all the generated label files (`.txt` files)
- Groups detections by date and label type
- Creates a line plot showing duration of each label category over time
- Helps identify patterns in the audio data (e.g., more pig activity on certain days)

## Input Structure

The script expects audio files organized in folders:
```
source_folder/
├── date1/
│   ├── recording1.wav
│   └── recording2.wav
└── date2/
    ├── recording3.wav
    └── recording4.wav
```

## Output

For each processed audio file, the script generates:
- **Audacity label file** (`filename.wav.txt`) - Timeline with labeled segments
- **Separate audio files** for each label (`filename_label_0.wav`, etc.)
- **Organized output folder** with descriptive naming

## Key Features

- **GPU acceleration** - Uses CUDA when available for faster processing
- **Batch processing** - Handles large audio files efficiently
- **Flexible configuration** - Easy to modify labels and processing parameters
- **Audio visualization** - Can generate square plot visualizations of classifications
- **Error handling** - Validates input parameters and handles edge cases

## Technical Details

- Uses librosa for audio processing
- Employs Microsoft's CLAP model for audio-text similarity
- Processes audio in overlapping frames to capture temporal patterns
- Exports results in formats compatible with Audacity and other audio tools

This tool is particularly useful for researchers analyzing animal vocalizations, environmental sounds, or any audio classification task where temporal segmentation is important.

## Citation

This project uses the CLAP model. Reference the original work:

**CLAP: Learning Audio Concepts from Natural Language Supervision**  
Elizalde, B., Deshmukh, S., Al Ismail, M., & Wang, H. (2023). *ICASSP 2023 – IEEE International Conference on Acoustics, Speech and Signal Processing*, 1–5.  

**Natural Language Supervision for General-Purpose Audio Representations**  
Elizalde, B., Deshmukh, S., & Wang, H. (2023). *arXiv:2309.05767*. [https://arxiv.org/abs/2309.05767](https://arxiv.org/abs/2309.05767)
