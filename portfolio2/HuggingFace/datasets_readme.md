# Big Data? 🤗 Datasets to the Rescue!

A comprehensive Jupyter notebook demonstrating how to efficiently work with large datasets using the Hugging Face Datasets library. This notebook shows you how to load, process, and stream massive datasets that would typically exceed your system's memory.

## Overview

This notebook tackles the challenge of working with big data in machine learning by leveraging the streaming capabilities of the 🤗 Datasets library. Instead of loading entire datasets into memory, you'll learn to process data efficiently through streaming, making it possible to work with multi-gigabyte datasets on standard hardware.

## What You'll Learn

- Loading large datasets efficiently without exhausting system memory
- Using dataset streaming to process data on-the-fly
- Measuring memory usage and performance metrics
- Applying transformations to streamed datasets (tokenization, shuffling)
- Splitting datasets for training and validation
- Combining multiple datasets through interleaving
- Working with compressed data formats (`.jsonl.zst`)

## Datasets Used

The notebook demonstrates techniques with several real-world datasets:

- **PubMed Abstracts** (19.5 GB): Medical research abstracts from 2020
- **FreeLaw Opinions**: Legal documents and court opinions
- **The Pile**: A large-scale diverse text dataset for language modeling

## Requirements

```bash
pip install datasets evaluate transformers[sentencepiece]
pip install zstandard
pip install psutil
```

## Key Features Demonstrated

### Memory Efficiency
- Standard loading: ~5.7 GB RAM for a 19.5 GB dataset
- Streaming mode: Minimal memory footprint regardless of dataset size

### Performance Metrics
- Dataset iteration speed: ~0.3 GB/s
- Processing 15.5 million examples efficiently
- Real-time memory monitoring with `psutil`

### Streaming Operations
- **Tokenization**: Apply transformers tokenizers to streaming data
- **Shuffling**: Randomize data order with configurable buffer size
- **Taking/Skipping**: Create train/validation splits without loading full dataset
- **Interleaving**: Combine multiple datasets seamlessly

## Usage Examples

### Load a Dataset with Streaming

```python
from datasets import load_dataset

dataset = load_dataset(
    "json", 
    data_files="path/to/data.jsonl.zst", 
    split="train", 
    streaming=True
)
```

### Apply Transformations

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"]))
```

### Create Train/Validation Splits

```python
shuffled = dataset.shuffle(buffer_size=10_000, seed=42)
train_dataset = shuffled.skip(1000)
validation_dataset = shuffled.take(1000)
```

## When to Use This Approach

- Working with datasets larger than your available RAM
- Processing data from remote sources without full downloads
- Prototyping with large datasets before committing to full processing
- Training models on-the-fly without intermediate storage
- Combining multiple large datasets

## Performance Considerations

- Streaming trades some processing speed for massive memory savings
- Buffer size affects shuffling randomness (larger = more random, but more memory)
- Network bandwidth impacts streaming performance with remote datasets
- Compressed formats (`.zst`) reduce download time and storage

## Additional Resources

- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets)
- [Streaming Guide](https://huggingface.co/docs/datasets/stream)
- [The Pile Dataset](https://pile.eleuther.ai/)

## Notes

- The notebook includes timing benchmarks to measure performance
- Memory usage is monitored throughout to demonstrate efficiency gains
- All examples use real, publicly available datasets
- Compressed data formats require the `zstandard` library

## License

This notebook demonstrates the use of various open datasets. Please refer to each dataset's individual license for usage terms.

---

**Pro Tip**: Start with streaming mode when exploring a new large dataset. You can always switch to full loading later if your workflow requires random access or if the dataset fits comfortably in memory.