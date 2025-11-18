# Adversarial Attack on Vision Models

Image-scaling based prompt injection attacks on multimodal LLMs. Because turns out, what you see isn't always what the model sees!!

## About the Project

This project demonstrates a subtle but powerful vulnerability: **hiding instructions in images that only become visible after downsampling**. Upload an innocent-looking photo, but after the model preprocesses it (scales it down), suddenly there's text telling it to do something else entirely.

## How It Works

The attack leverages a fundamental property of image preprocessing:

1. **Bilinear Embedding**: Text is carefully embedded in the high-resolution image's dark regions using mathematically optimized pixel modifications
2. **Invisible at Full-Res**: The changes are subtle enough to be invisible to the human eye at original resolution
3. **Visible After Downsampling**: When the model downsamples the image (as most vision APIs do for efficiency), the hidden text becomes readable

The math behind it involves solving an optimization problem for each 4x4 pixel block, modifying only dark pixels (where changes are hardest to see) to create text that "materializes" during downsampling.

## Project Structure

```
├── app.py                          # Main Streamlit web interface
├── adversarial_img_gen.py          # Core embedding algorithm
├── target_img_gen.py               # Text placement and optimization
├── utils.py                        # Color space conversions, masks
├── vision_model_test_scripts/      # Test scripts for various models
│   ├── openai_test.py             # GPT-4o/4o-mini testing
│   ├── moondream_test.py          # Moondream model testing
│   ├── smol_vlm_test.py           # SmolVLM testing
│   └── clip_test.py               # CLIP model testing
├── docs/
│   ├── impactAssessment.md        # Analysis of potential real-world impact
│   └── responsibleDisclosure.md   # Disclosure approach and academic context
├── adversarial_images/             # Example adversarial images
├── attack_images/                  # Screenshots of successful attacks
└── decoy_images/                   # Sample base images for embedding
```

## Features

### Interactive Web App (Streamlit)

- **Attack Tab**: Test adversarial images on local models (Moondream, SmolVLM, OpenAI GPT-4)
- **How It's Done Tab**: Step-by-step image generation with real-time metrics
  - Upload decoy image
  - Enter hidden text
  - See coverage heatmaps showing embeddable regions
  - Auto-calculated optimal font size and placement
  - Generate adversarial image with quality metrics (MSE/PSNR)
- **Defense Tab**: Simple but effective defense. Preview the downsampled version before processing
- **Attacking Commercial Models**: Gallery of successful attacks on GPT-4o, Cursor Codex, and Gemini
- **Impact Assessment**: Built-in documentation of risks and implications
- **Responsible Disclosure**: Academic context and ethical considerations

### Successful Attacks Demonstrated

- **ChatGPT (GPT-4o/4o-mini)**: Made it write poems, dark stories, and roleplay as a bird
- **Cursor (GPT-5.1-Codex-Mini)**: Tricked into creating potentially malicious files
- **Gemini (Google AI Mode)**: Replaced verbs with "chirping" based on hidden instructions
- **Open Source Models**: Moondream, SmolVLM also vulnerable

## Getting Started

### Prerequisites

- Python 3.12+
- For full functionality, you'll need:
  - OpenAI API key (for GPT-4 testing)
  - GPU recommended for local models (Moondream, SmolVLM)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/adversarial-attack-on-vision-models.git
cd adversarial-attack-on-vision-models

# Install uv if you haven't already
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install core dependencies (enough for image generation)
uv sync

# Optional: Install extras based on what you need
# For OpenAI API testing only:
uv sync --extra openai

# For local models (CPU version):
uv sync --extra models

# For local models with GPU support (install torch with CUDA first):
pip install torch --index-url https://download.pytorch.org/whl/cu121
uv sync --extra models

# For everything:
uv sync --extra all

# Set up API keys (optional, for commercial model testing)
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Running the App

```bash
uv run streamlit run app.py
```

The web interface will open in your browser. Try uploading a high-resolution square image (e.g., 4368×4368) and embedding some text to see the attack in action.

### Tips for Best Results

- Use square images divisible by 4 (1024×1024, 4368×4368). Higher Resolution = Better Results.
- Images with larger dark areas work best (more space to hide text)
- Keep embedded text relatively short for higher quality
- Adjust Dark Fraction slider if needed (higher = more editable pixels but potentially more visible)

## The Defense

Show users the downsampled version before processing! If users can see what the model sees, they'll spot the hidden text.

Google's Gemini already does this (showing a preview), though UI constraints (tiny preview images) can still let some attacks through.

The real lesson: secure AI != secure model; secure AI = secure orchestration

## Why This Matters

This isn't about model weakness. The models are doing their job correctly. It's about:

1. **Visibility Gap**: Users see one thing, models see another
2. **Preprocessing Opacity**: Downsampling/preprocessing happens invisibly to users
3. **Agentic Risk**: Models with tool access (email, calendar, file systems) can act on hidden instructions
4. **Persistent Attack Surface**: As long as images get preprocessed, this vulnerability exists

Even as models evolve, UI constraints, mobile apps, and backend services will continue to resize images. The attack generalizes across platforms and downsampling algorithms.

The goal: highlight a structural vulnerability in multimodal AI systems and demonstrate that preprocessing transparency matters.

## Further Reading

- [Impact Assessment](docs/impactAssessment.md) 
- [Responsible Disclosure](docs/responsibleDisclosure.md) 
