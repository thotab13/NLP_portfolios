# Optimizing Explicit Reasoning Traces in Gemma 3 using Tunix and GRPO

Evolving Gemma 3 1B into a Structured Reasoning Engine using the Tunix Framework and Multi-Objective Rewards

**Google Tunix Hackathon Submission** | January 2026

## Links

- 📊 **Training Results**: [Weights & Biases Dashboard](https://wandb.ai/thotab15-w-rzburg/tunix/table?nw=nwuserthotab15)
- 📝 **Competition Writeup**: [Kaggle Writeup](https://kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1767560834117)
- 🎥 **Video Presentation**: [YouTube](https://youtu.be/JusdhgNYGus)
- 💻 **Code Implementation**: [Kaggle Notebook](https://www.kaggle.com/code/madasaisurya/notebook443b76c5cf)

## Overview

This project addresses a critical limitation in modern language models: while they can produce correct answers to mathematical problems, they often fail to explicitly show their reasoning process. This limits interpretability, educational value, and error analysis capabilities.

Our solution transforms **Gemma 3 1B-IT** into a model that consistently generates structured, step-by-step mathematical reasoning using **Group Relative Policy Optimization (GRPO)** through Google's Tunix framework. Rather than relying on supervised chain-of-thought annotations, we induce reasoning quality through reinforcement learning with a carefully designed multi-component reward function.

## Key Innovation

**Reasoning transparency as a trainable behavior** — not an emergent side effect. By combining structured prompting with relative reinforcement learning, the model learns to externalize its reasoning process without requiring supervised chain-of-thought labels.

## Model Architecture

### Base Model
- **Model**: Gemma 3 1B-IT (instruction-tuned)
- **Adaptation**: Low-Rank Adaptation (LoRA)
- **Training Strategy**: LoRA parameters updated, base weights frozen
- **Reference Model**: Frozen copy maintained for KL-divergence penalties

### Why LoRA?
Enables efficient specialization while preserving general language competence and ensuring training stability.

## Dataset: SVAMP

**Simple Variations on Arithmetic Math Problems** — designed to test compositional reasoning through subtle linguistic and numerical variations.

### Dataset Characteristics
- Natural-language arithmetic word problems
- Deterministic numeric ground-truth answers
- Multi-step reasoning requirements
- 80/20 train-test split
- Sorted by equation complexity for curriculum learning

### Why SVAMP?
Perfect for reasoning alignment because correct solutions require explicit decomposition into intermediate steps rather than surface-level pattern matching.

## Training Methodology

### 1. Structured Output Format

Strict schema enforced throughout training using `<reasoning>` and `<answer>` tags to separate the step-by-step explanation from the final numeric result.

**Benefits:**
- Precise reward computation
- Prevents reasoning leakage into final answer
- Ensures consistent, machine-parseable outputs

### 2. Group Relative Policy Optimization (GRPO)

Reinforcement learning method that compares multiple candidate outputs for the same prompt.

**Process per training example:**
1. Generate 4 candidate responses
2. Evaluate each using multiple reward components
3. Normalize rewards relative to the group
4. Update policy to favor higher-quality reasoning traces
5. Apply KL-divergence penalty to prevent excessive drift

**Training Configuration:**
- Generations per prompt: 4
- KL coefficient (β): 0.04
- Clipping parameter (ε): 0.2
- Total training steps: 2000
- Hardware: TPU v5e-8

## Custom Reward Function

Seven-component reward function decomposing reasoning quality into distinct, measurable signals:

### 1. Exact Format Compliance
Rewards strict adherence to `<reasoning>` and `<answer>` tag structure.

### 2. Approximate Format Compliance
Provides partial credit for near-compliant outputs, supporting gradual alignment.

### 3. Numeric Answer Correctness
Extracts numeric value from `<answer>` block and compares against ground truth.

### 4. Flexible Numeric Validation
Handles minor formatting variations in numeric expressions.

### 5. Reasoning Coherence
Rewards logical progression, stepwise explanations, and clear transitions.

### 6. Meaningful Reasoning Length
Encourages sufficiently detailed explanations while discouraging trivial outputs.

### 7. Algebraic Notation Usage
Rewards explicit arithmetic expressions and equation-based reasoning.

**Key Design Principle:** All components applied relatively within each generation group, teaching the model what distinguishes high-quality reasoning from weaker alternatives.

## Results

### Quantitative Performance (SVAMP Test Set)

| Metric | Before GRPO | After GRPO | Improvement |
|--------|-------------|------------|-------------|
| **Accuracy** | 54.29% | 67.14% | +12.85% |
| **Partial Accuracy** | 54.29% | 67.14% | +12.85% |
| **Format Compliance** | 85.71% | 98.57% | +12.86% |

### Key Findings
- Structured reasoning improves both interpretability **and** task accuracy
- Format compliance reaches near-perfect levels through reward design alone
- Accuracy improvement demonstrates correlation between explicit reasoning and correctness

### Qualitative Improvements

Clear behavioral changes observed:

✅ Consistent production of explicit intermediate reasoning steps  
✅ Reliable adherence to structured format  
✅ Sequential explanation of arithmetic operations  
✅ Transparent error sources when mistakes occur  
✅ Student-like "showing work" behavior

## Project Structure

```
├── svamp-complete.ipynb        # Dataset processing and analysis
├── trained-gemma3.ipynb        # Training implementation and results
└── README.md                   # This file
```

## Reproduction

### Requirements
- Google Tunix framework
- TPU v5e-8 or equivalent
- Gemma 3 1B-IT model access
- SVAMP dataset

### Quick Start
1. Load the training notebook: `trained-gemma3.ipynb`
2. Configure TPU runtime
3. Run all cells to reproduce training
4. Evaluate on SVAMP test set

## Technical Contributions

1. **Multi-Component Reward Design**: Novel seven-component reward function for reasoning quality
2. **Format Enforcement via RL**: Structural constraints through reward design without hard-coding
3. **Relative Optimization**: Group-based normalization avoiding brittle absolute thresholds
4. **Curriculum Learning**: Complexity-sorted training for stable reasoning development

## Impact & Applications

### Educational Use Cases
- Transparent step-by-step problem solving
- Error analysis and debugging support
- Mathematics tutoring systems

### Research Applications
- Interpretability and explainability research
- Reasoning alignment studies
- Reinforcement learning for structured generation

### Production Systems
- Verifiable mathematical reasoning
- Audit trails for AI decision-making
- Human-in-the-loop validation workflows

## Limitations & Future Work

### Current Limitations
- Limited to arithmetic/mathematical reasoning
- Constrained to SVAMP problem complexity
- 1B parameter model size

### Future Directions
- Extension to other reasoning domains (logic, physics, code)
- Scaling to larger model sizes (7B, 27B)
- Multi-modal reasoning with visual inputs
- Cross-lingual reasoning alignment

## Competition Context

**Google Tunix Hackathon**: Train a model to show its work

This submission directly addresses the competition's core objective by demonstrating that reasoning transparency can be reliably trained through reinforcement learning, achieving high format compliance and improved task performance without supervised chain-of-thought labels.

## Authors

- **Shehariyar Shaikh** - [@shehariyarshaikh](https://www.kaggle.com/shehariyarshaikh)
- **Thota Bhuvana Chandra** - [@thotabhuvanachandra](https://www.kaggle.com/thotabhuvanachandra)
- **Mada Sai Surya** - [@madasaisurya](https://www.kaggle.com/madasaisurya)
- **Rachitha C** - [@rachitha02](https://www.kaggle.com/rachitha02)

## License

This project is released under the [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

## Citation

```bibtex
@misc{shaikh2026gemma3reasoning,
  author = {Shehariyar Shaikh and Thota Bhuvana Chandra and Mada Sai Surya and Rachitha C},
  title = {Optimizing Explicit Reasoning Traces in Gemma 3 using Tunix and GRPO},
  year = {2026},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1767560834117}}
}
```

## Acknowledgments

- **Google**: For the Tunix framework and hackathon
- **Gemma Team**: For the base model
- **SVAMP Dataset**: For the reasoning benchmark

---

*Demonstrating that reasoning transparency is trainable, not emergent.*