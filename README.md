# Korean Speculative Decoding H100 Research

## 🎯 Project Overview

This project investigates the counterintuitive phenomenon where smaller language models can outperform larger ones in speculative decoding scenarios, specifically focusing on Korean language processing using H100 GPUs.

## 🔬 Research Question

**"Can smaller models achieve better efficiency and performance than larger models in Korean language speculative decoding?"**

### Key Findings from M3 Experiments
- 1.3B model outperformed 2.7B model in certain scenarios
- Speculative decoding showed significant speed improvements
- Korean language exhibited unique characteristics

## 🏗️ Project Structure

```
korean_speculative_decoding_h100/
├── src/           # Core source code
├── models/        # Model storage (not in git)
├── configs/       # Configuration files
├── scripts/       # Execution scripts
├── data/          # Dataset storage
├── results/       # Experiment results
├── notebooks/     # Jupyter notebooks
└── paper/         # Research paper
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models
```bash
# Download essential models for research
python scripts/setup/download_models.py --category size_comparison
```

### 3. Run Basic Experiment
```bash
# Reproduce M3 findings on H100
python scripts/experiments/run_m3_reproduction.py
```

## 📊 Experiment Categories

### 1. Size Comparison Study
- Reproduce M3 findings with updated models
- Compare 117M → 70B parameter ranges
- Analyze performance vs efficiency trade-offs

### 2. Korean Language Optimization
- Korean-specific model evaluation
- Language-aware speculative decoding
- Cultural and linguistic factor analysis

### 3. Scaling Analysis
- Performance scaling with model size
- H100 utilization optimization
- Memory and compute efficiency

## 🔧 Hardware Requirements

- **GPU**: NVIDIA H100 (80GB recommended)
- **RAM**: 128GB+ system memory
- **Storage**: 2TB+ for models and data
- **CUDA**: 11.8+ or 12.0+

## 📈 Expected Outcomes

1. **Academic**: Novel insights into model scaling and efficiency
2. **Practical**: Optimized Korean language AI systems
3. **Industrial**: Cost-effective deployment strategies

## 🤝 Contributing

Please read our [contribution guidelines](docs/CONTRIBUTING.md) before submitting PRs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citations

If you use this work, please cite:
```
@article{korean_speculative_decoding_h100,
  title={Korean Speculative Decoding on H100: When Smaller Models Excel},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Institution**: Your Institution

---

**🇰🇷 한국어 문서**: [README_KR.md](docs/README_KR.md)
