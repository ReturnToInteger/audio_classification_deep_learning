
import re
from msclap import CLAP
import numpy as np
import soundfile as sf
import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
import random
import argparse

DEFAULT_CONFIG = {
    "labels": ["background noise", "brief peak event, like a clap or click", "pig sneeze", "pig oink"],
    "source": "E:/Documents/Projects/pig_noises/src/drive/pig_sounds",
    "output_folder": "src/colab/local/",
    "file_per_folder": 3,
    "batch_size": 70,
    "sample_rate": 16000,
    "frame_s": 1,
    "hop_s": 0.5,
    "temp_folder": "temp/frames"
}

def save_frames(frames, path, sr):
    """
    Saves audio frames to a specified folder.

    Args:
        frames (np.ndarray): Array of audio frames.
        path (str): The directory to save the frames to.
        sr (int): The sampling rate of the audio.
    """
    os.makedirs(path, exist_ok=True)
    outputs=[]
    for i, frame in enumerate(frames.T):
        output_filename = os.path.join(path, f'frame_{i:04d}.wav')
        outputs.append(output_filename)
        sf.write(output_filename, frame, sr)
    return outputs

def id_to_time(idx, hop_length, frame_length, sr):
    """ Convert frame index to start and end times in seconds."""
    start = idx * hop_length
    end = start + frame_length
    return start / sr, end / sr

def calc_regions(ids : list,hop_length : float,frame_length : float) -> list[tuple[float, float, int]]:
    """ 
    Calculate regions from frame IDs.
    Returns a list of tuples (start, end, label) for each region.
    Each region is defined by the start and end sample indices and the label ID.
    The start and end are calculated based on the hop length and frame length.
    The last region extends to the end of the audio.

    ids: array of frame labels.

    hop_length: number of samples between frames.

    frame_length: number of samples in each frame.

    Returns: list of (start, end, label) tuples in seconds.
    """
    regions=[]
    start=0
    for i in range(len(ids)-1):
        if ids[i]!=ids[i+1]:  
            #end
            end=i*hop_length+frame_length
            regions.append((start,end,ids[i]))
            #start
            start=(i+1)*hop_length
    end=len(ids)*hop_length+frame_length
    regions.append((start,end,ids[-1]))
    return regions

# Save labels to import in audacity
def save_audacity_labels(regions, sr, filename):
    """
    regions: list of (start_sample, end_sample, label) tuples
    sr: sample rate
    filename: output .txt path
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for start, end, label in regions:
            start_sec = start / sr
            end_sec = end / sr
            f.write(f"{start_sec:.6f}\t{end_sec:.6f}\t{label}\n")

class AudioFrameExtractor:
    """Extracts audio frames from a file and saves them to a specified folder."""
    input_filepath: str
    output_folder: str
    sr: int | float
    frame_s: float
    hop_s: float
    frames: np.ndarray
    @property
    def frame_size(self):
        return int(self.sr * self.frame_s )
    @property
    def hop_length(self):
        return int(self.sr * self.hop_s)
    def __init__(self, input_filepath: str, output_folder: str = "/content/frames", sr: int | float = 16000, frame_s: float = 3, hop_s: float = 1.5 ) -> None:
        """Initializes the AudioFrameExtractor with the input file path, output folder, sample rate, frame size, and hop size."""
        self.input_filepath=input_filepath
        self.output_folder=output_folder
        self.sr=sr
        self.frame_s=frame_s
        self.hop_s=hop_s
        self.y, sr = librosa.load(self.input_filepath, sr=self.sr)
    def process_frames(self) -> list[str]:
        """Processes the audio file into frames and saves them to the output folder."""
        frames = librosa.util.frame(self.y, frame_length=self.frame_size, hop_length=self.hop_length)
        frame_paths=save_frames(frames, self.output_folder, self.sr)
        return frame_paths
    def slice_audio(self, start: float, end: float) -> np.ndarray:
        """Slices the audio from start to end time in seconds."""
        start_sample = int(start * self.sr)
        end_sample = int(end * self.sr)
        return self.y[start_sample:end_sample]

def batch_process_audio(model : CLAP, file_paths : list[str], batch_size : int, text_embeddings):
    """
    From a list of audio file paths, compute the similarity with text embeddings in batches.
    Args:
        model (CLAP): The CLAP model instance.
        file_paths (list[str]): List of audio file paths.
        batch_size (int): Number of files to process in each batch.
        text_embeddings: Precomputed text embeddings for similarity comparison.
    Returns:
        np.ndarray: A 2D array of similarities with shape (num_frames, num_text_embeddings).
    """
    similarities=[]
    for j,i in enumerate(range(0, len(file_paths), batch_size)):
        batch_paths = file_paths[i:i+batch_size]
        audio_embeddings = model.get_audio_embeddings(batch_paths)
        # Compute similarity between audio and text embeddings
        sim=model.compute_similarity(audio_embeddings, text_embeddings)
        similarities.append(sim.cpu().detach().numpy())
        print(f"{j}: {similarities[j].min()}, {similarities[j].max()}")
    return np.vstack(similarities)

def get_folders(path):
    """Return a list of folder names in the given path."""
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def get_files(path):
    """Return a list of file names in the given path."""
    return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]

def plot_image_square(label_ids : np.ndarray, no_of_labels : int,output : str | None = None):
    """ Plot a square image from label IDs."""
    im_array= (label_ids/(no_of_labels-1)*255).astype(int)
    side_length= int(np.sqrt(len(im_array)))
    im_array=np.reshape(im_array[:side_length**2],(side_length,side_length))

    plt.imshow(im_array)
    plt.axis('off')
    if output:
        if not output.endswith(".png"):
            output += ".png"
        plt.savefig(output, bbox_inches='tight', pad_inches=0)
    plt.show()

def save_similarity_to_csv(similarity_array : np.ndarray,labels, output):
    """ Save similarity array to a CSV file."""
    df=pd.DataFrame(similarity_array, columns=labels, index= [f"frame_{i:04d}" for i in range(similarity_array.shape[0])])
    if not output.endswith(".csv"):
        output += "_sim.csv"
    df.to_csv(output, sep=";",decimal=",", float_format="%.4f")

_illegal_chars = r'[<>:"/\\|?*]'

def sanitize_filename(name, replace_with="_"):
    """
    Replace illegal characters in a filename with a safe substitute.

    Example:
    -------
    >>> sanitize_filename("bad<name>.txt")
    'bad_name_.txt'
    """

    return re.sub(_illegal_chars, replace_with, name)

def process_single_audio_file_in_batch(clap_model : CLAP, folder : str, file : str, batch_size : int, labels : list[str],
                                       sample_rate : int, frame_s : float, hop_s : float, out_folder : str, 
                                       text_embeddings, temp_folder : str):
    """
    Process a single audio file, extract frames, compute similarities, and save results.
    Args:
        clap_model (CLAP): The CLAP model instance.
        folder (str): The folder containing the audio file.
        file (str): The name of the audio file.
        batch_size (int): Number of frames to process in each batch.
        labels (list[str]): List of labels for classification.
        sample_rate (int): Sample rate for audio processing.
        frame_s (float): Frame size in seconds.
        hop_s (float): Hop size in seconds.
        out_folder (str): Output folder to save results.
        text_embeddings: Precomputed text embeddings for similarity comparison.
        temp_folder (str): Temporary folder for intermediate files.
    """
    extractor=AudioFrameExtractor(os.path.join(folder,file),temp_folder, 
                                    sr=sample_rate, frame_s=frame_s, hop_s=hop_s)
    frame_paths=extractor.process_frames()
    print(f"Framed {file}")

    similarities = batch_process_audio(clap_model, frame_paths, batch_size, text_embeddings)

    label_ids= similarities.argmax(axis=1)
    regions=calc_regions(label_ids,extractor.hop_length,extractor.frame_size)
    output_file = os.path.join(out_folder, f"{file}.txt")
    save_audacity_labels(regions,extractor.sr,output_file)
    print(f"Labels saved to {output_file}")

    for label_idx, label_name in enumerate(labels):
        label_path = os.path.join(out_folder, f"{file}_label_{label_idx}.wav")
        with sf.SoundFile(label_path, 'w', samplerate=extractor.sr, channels=1) as f:
            for start, end, lbl in regions:
                if lbl != label_idx:
                    continue
                frame_slice = extractor.y[start : end]
                f.write(frame_slice)
        print(f"Saved labeled audio for label {label_idx} to {label_path}")
    
    del extractor
    del frame_paths

def process_audio_folders(config : argparse.Namespace):
    """
    Process audio files and extract labeled segments for classification.

    Detailed description:
    -------------------
    Splits audio into frames, computes embeddings, and saves results 
    to the specified output folder. Uses a temporary folder for 
    intermediate data.

    Parameters:
    ----------
    config : argparse.Namespace
        Configuration object containing:
        - labels: List of labels for classification.
        - source: Source folder containing audio files. The folder structure should be:
          - source/
            - date1/
              - file1.wav
              - file2.wav
            - date2/
              - file1.wav
              - file2.wav
        - output_folder: Folder to save processed results.
        - file_per_folder: Number of files to process per folder.
        - batch_size: Batch size for processing audio files.
        - sample_rate: Sample rate for audio processing.
        - frame_s: Frame size in seconds.
        - hop_s: Hop size in seconds.
        - temp_folder: Temporary folder for intermediate files.

    Returns:
    -------
    None

    Example:
    -------
    >>> config = argparse.Namespace(**DEFAULT_CONFIG)
    >>> process_audio(config)
    """

    print("Processing...")
    print(f"Labels: {config.labels}")
    print(f"Source: {config.source}")
    labels_str= "__".join(config.labels).replace(" ", "_")
    top_out_folder = sanitize_filename(f"output_{config.frame_s}_{config.hop_s}_{labels_str}")
    out_folder = os.path.join(config.output_folder, top_out_folder)
    print(f"Output: {out_folder}")
    print(f"Files per folder: {config.file_per_folder}, Batch size: {config.batch_size}")
    print(f"Frame: {config.frame_s}s, Hop: {config.hop_s}s")

    os.makedirs(out_folder, exist_ok=True)
    clap_model = CLAP(version = '2023', use_cuda=True)
    print("Model loaded.")
    text_embeddings = clap_model.get_text_embeddings(config.labels)
    print(text_embeddings.device)
    for f in get_folders(config.source):
      folder=os.path.join(config.source, f)
      print(f"Processing folder: {folder}")
      files_all=get_files(folder)
      files_sample=random.sample(files_all,min(config.file_per_folder,len(files_all)))
      for file in files_sample:
        print(f"Processing file: {file}")
        process_single_audio_file_in_batch(clap_model, folder, file, config.batch_size, config.labels,
                                           config.sample_rate, config.frame_s, config.hop_s, out_folder, 
                                           text_embeddings, config.temp_folder)





def validate_config(args):
    if not isinstance(args.labels, list) or not all(isinstance(l, str) for l in args.labels):
        raise ValueError("labels must be a list of strings.")
    if not isinstance(args.source, str):
        raise ValueError("source must be a string path.")
    if not isinstance(args.output_folder, str):
        raise ValueError("output_folder must be a string path.")
    if not isinstance(args.file_per_folder, int) or args.file_per_folder <= 0:
        raise ValueError("fie_per_folder must be a positive integer.")
    if not isinstance(args.batch_size, int) or args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(args.sample_rate, int) or args.sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if not isinstance(args.frame_s, (int, float)) or args.frame_s <= 0:
        raise ValueError("frame_s must be a positive number.")
    if not isinstance(args.hop_s, (int, float)) or args.hop_s <= 0:
        raise ValueError("hop_s must be a positive number.")    
    if not isinstance(args.temp_folder, str):
        raise ValueError("temp_folder must be a string path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio processing script with defaults from DEFAULT_CONFIG")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_CONFIG["labels"], help="List of labels")
    parser.add_argument("--source", default=DEFAULT_CONFIG["source"], help="Source folder")
    parser.add_argument("--output_folder", default=DEFAULT_CONFIG["output_folder"], help="Output folder")
    parser.add_argument("--file_per_folder", type=int, default=DEFAULT_CONFIG["file_per_folder"], help="Files per folder")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_CONFIG["sample_rate"], help="Audio sample rate")
    parser.add_argument("--frame_s", type=float, default=DEFAULT_CONFIG["frame_s"], help="Frame size in seconds")
    parser.add_argument("--hop_s", type=float, default=DEFAULT_CONFIG["hop_s"], help="Hop size in seconds")
    parser.add_argument("--temp_folder", default=DEFAULT_CONFIG["temp_folder"], help="Temporary folder")

    args = parser.parse_args()

    # Ensure output and temp folders exist
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.temp_folder, exist_ok=True)


    # Validate final config
    validate_config(args)
        
    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    process_audio_folders(args)
