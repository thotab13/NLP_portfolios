# Offline Poem Generator 🎭✨

An interactive Python application that uses GPT-2 to generate creative poetry locally on your machine. No internet connection required after initial model download!

## Overview

This Jupyter notebook demonstrates how to build a fully offline AI-powered poetry generator using Hugging Face's Transformers library. The application prompts users for three words and weaves them into unique, creative poems using the GPT-2 language model.

## Features

- **Fully Offline**: Once the model is downloaded, no internet connection needed
- **GPU Acceleration**: Automatically detects and uses CUDA-enabled GPUs for faster generation
- **Interactive CLI**: Simple command-line interface for easy interaction
- **Creative Output**: Uses advanced sampling techniques for diverse, coherent poetry
- **Customizable**: Easy-to-modify parameters for different creative styles

## Requirements

```bash
pip install transformers torch
```

### Hardware
- **CPU**: Works on any modern CPU (slower generation)
- **GPU**: CUDA-enabled GPU recommended for faster generation
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: ~2GB for model files

## How It Works

The generator uses a two-line prompt strategy to guide GPT-2:

1. You provide three words
2. The system creates a poetic starter prompt incorporating your words
3. GPT-2 continues the poem with creative text generation
4. Advanced sampling ensures coherent, diverse output

### Generation Parameters

The notebook uses optimized parameters for creative writing:

- **Temperature (0.8)**: Balances creativity with coherence
- **Top-k (50)**: Samples from the top 50 most likely next words
- **Top-p (0.92)**: Nucleus sampling for diverse vocabulary
- **No-repeat n-grams (3)**: Prevents repetitive loops

## Usage

### Running the Notebook

1. Open the notebook in Jupyter or Google Colab
2. Run the setup cell to load the model
3. Enter three words when prompted
4. Enjoy your generated poem!
5. Type `exit` or `quit` to end the session

### Example Interaction

```
Word 1: dream
Word 2: bike
Word 3: sweet

Generating...

====================
Beneath the bike, a dream did stray,
Lost in a sweet at the end of the day.
"I'm the one who's done it and I'm the guy who's kept it going,
I'm just so proud of it, and it was a dream come true..."
====================
```

## Model Information

- **Model**: GPT-2 (124M parameters)
- **Size**: ~548MB
- **Provider**: Hugging Face Transformers
- **License**: MIT (OpenAI)

## Performance

- **GPU Mode**: ~2-5 seconds per poem (CUDA)
- **CPU Mode**: ~10-30 seconds per poem
- **First Run**: Additional time for model download (~1-2 minutes)

## Technical Details

### Model Loading
```python
generator = pipeline("text-generation", model="gpt2", device=device)
```

The script automatically detects available hardware:
- `device=0`: GPU (CUDA) if available
- `device=-1`: CPU fallback

### Prompt Engineering

The generator uses a structured prompt format:
```python
prompt = f"Beneath the {word2}, a {word1} did stray,\n"
         f"Lost in a {word3} at the end of the day.\n"
```

This provides a poetic structure that GPT-2 naturally continues.

### Randomization

Each generation uses the system clock as a seed for variety:
```python
set_seed(int(time.time()))
```

## Customization

### Adjusting Output Length

Modify the `new_tokens` parameter:
```python
def generate_poem(generator, word1, word2, word3, new_tokens=150):
    # Increase/decrease for longer/shorter poems
```

### Changing Creative Style

Experiment with these parameters:
- `temperature`: Higher = more random (0.7-1.2)
- `top_k`: Size of candidate pool (30-100)
- `top_p`: Nucleus sampling threshold (0.85-0.98)

### Custom Prompts

Replace the prompt template in `generate_poem()`:
```python
prompt = f"Your custom prompt with {word1}, {word2}, {word3}"
```

## Troubleshooting

### Out of Memory Errors
- Reduce `new_tokens` parameter
- Use CPU mode instead of GPU
- Close other applications

### Slow Generation
- Enable GPU acceleration
- Reduce `new_tokens`
- Use a smaller model variant

### Model Download Issues
- Check internet connection on first run
- Manually download model from Hugging Face Hub
- Verify storage space (need ~2GB free)

## Use Cases

- Creative writing inspiration
- Poetry education and exploration
- Language model demonstration
- Offline AI experimentation
- Fun interactive art project

## Limitations

- GPT-2 is not specifically trained on poetry (outputs vary in quality)
- May occasionally generate nonsensical or incomplete text
- Limited understanding of poetic structure and meter
- No rhyme scheme guarantee
- Outputs can be unpredictable

## Future Improvements

- Add fine-tuning on poetry datasets
- Implement rhyme scheme detection/enforcement
- Support for different poetic forms (haiku, sonnet, limerick)
- Save favorite poems to file
- Web interface with Flask/Streamlit
- Use larger models (GPT-2 Medium/Large) for better quality

## Educational Value

This notebook demonstrates:
- Using pre-trained language models locally
- GPU vs CPU inference
- Text generation sampling strategies
- Prompt engineering techniques
- Interactive Python applications
- Resource management and optimization

## License

This code is provided as educational material. The GPT-2 model is released under the MIT license by OpenAI.

## Acknowledgments

- **OpenAI**: For GPT-2 model
- **Hugging Face**: For Transformers library
- **PyTorch**: For deep learning framework