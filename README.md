# Orii-Demo-2

# Orii-O1: Advanced Offline Multimodal LLM

Orii-O1 is a complete, production-ready Large Language Model with advanced image generation capabilities, designed to run entirely offline without internet connectivity. It features human-like text generation, high-quality image synthesis, and multimodal understanding.

## ✨ Features

- **🧠 Advanced Text Generation**: Human-like responses with conversational patterns
- **🎨 High-Quality Image Generation**: Diffusion-based image synthesis from text prompts
- **👁️ Multimodal Understanding**: Process both text and images simultaneously
- **🔒 Completely Offline**: No internet connection required after setup
- **⚡ Optimized Performance**: GPU acceleration with mixed precision support
- **🌐 Web Interface**: User-friendly chat interface and API
- **🛡️ Built-in Safety**: Content filtering and safety mechanisms
- **📱 Easy Deployment**: Docker support and custom domain deployment

## 🏗️ Architecture

Orii-O1 combines three core components:

1. **Transformer Model**: Custom architecture with RoPE, SwiGLU activation, and attention optimizations
2. **Diffusion Model**: U-Net based image generator with text conditioning
3. **Multimodal Fusion**: Cross-attention mechanisms for text-image understanding

## 📋 Requirements

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080 or better recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space for models and data
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, or macOS

### Software Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Docker (optional, for deployment)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/orii-o1.git
cd orii-o1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. Prepare Training Data

Create your training data structure:

```
data/
├── train/
│   ├── text/
│   │   ├── conversations.jsonl
│   │   └── documents.jsonl
│   └── images/
│       ├── image1.jpg
│       ├── image1.txt  # Caption file
│       ├── image2.png
│       └── image2.txt
└── validation/
    ├── text/
    └── images/
```

**Text Data Format (JSONL):**
```json
{"text": "Hello! How can I help you today?"}
{"conversation": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is artificial intelligence..."}]}
```

**Image Data:**
- Place images in the images folder
- Create corresponding .txt files with captions
- Supported formats: JPG, PNG, GIF

### 3. Training

#### Basic Training
```bash
python scripts/train.py \
    --data_dir ./data/train \
    --model_size medium \
    --batch_size 4 \
    --num_epochs 10 \
    --learning_rate 5e-4 \
    --mixed_precision fp16 \
    --use_wandb
```

#### Advanced Training Options
```bash
python scripts/train.py \
    --data_dir ./data/train \
    --model_size large \
    --mode multimodal \
    --batch_size 8 \
    --num_epochs 20 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --warmup_steps 2000 \
    --gradient_accumulation_steps 4 \
    --mixed_precision fp16 \
    --checkpoint_dir ./models/checkpoints \
    --save_every 1000 \
    --eval_every 500 \
    --use_wandb \
    --project_name my-orii-o1
```

#### Resume Training
```bash
python scripts/train.py \
    --resume_from ./models/checkpoints/latest_checkpoint.pt \
    --data_dir ./data/train \
    --num_epochs 10
```

### 4. Inference

#### Command Line Interface
```bash
# Text generation
python scripts/inference.py \
    --model_path ./models/checkpoints/best_model.pt \
    --mode text \
    --prompt "Hello, how are you today?"

# Image generation
python scripts/inference.py \
    --model_path ./models/checkpoints/best_model.pt \
    --mode image \
    --prompt "A beautiful sunset over the ocean"

# Interactive chat
python scripts/inference.py \
    --model_path ./models/checkpoints/best_model.pt \
    --mode chat

# Multimodal (text + image input)
python scripts/inference.py \
    --model_path ./models/checkpoints/best_model.pt \
    --mode multimodal \
    --prompt "What do you see in this image?" \
    --image_path ./path/to/image.jpg
```

### 5. Web Interface

#### Start Web Server
```bash
python web_interface/app.py \
    --model_path ./models/checkpoints/best_model.pt \
    --host 0.0.0.0 \
    --port 5000
```

#### Access the Interface
- Open your browser and go to `http://localhost:5000`
- Use the chat interface for text conversations
- Upload images for multimodal interactions
- Generate images from text prompts

## 🐳 Docker Deployment

### Build Docker Image
```bash
# Build the image
docker build -t orii-o1:latest -f deployment/docker/Dockerfile .

# Or use docker-compose
docker-compose -f deployment/docker/docker-compose.yml up --build
```

### Run Container
```bash
docker run -d \
    --name orii-o1 \
    --gpus all \
    -p 5000:5000 \
    -v ./models:/app/models \
    -v ./data:/app/data \
    orii-o1:latest
```

## 🌐 Custom Domain Deployment

### 1. Nginx Configuration
```bash
# Copy nginx config
sudo cp deployment/nginx/nginx.conf /etc/nginx/sites-available/orii-o1
sudo ln -s /etc/nginx/sites-available/orii-o1 /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 2. Systemd Service
```bash
# Install systemd service
sudo cp deployment/systemd/orii-o1.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable orii-o1
sudo systemctl start orii-o1
```

### 3. SSL Certificate (Let's Encrypt)
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

## 📊 Model Configurations

### Small Model (2B parameters)
- **Use Case**: Development, testing, resource-constrained environments
- **VRAM**: 4-6GB
- **Speed**: Fast inference
```bash
--model_size small
```

### Medium Model (7B parameters)
- **Use Case**: Balanced performance and resource usage
- **VRAM**: 8-12GB
- **Speed**: Good balance
```bash
--model_size medium
```

### Large Model (13B parameters)
- **Use Case**: Best quality, research, production
- **VRAM**: 16-24GB
- **Speed**: Slower but highest quality
```bash
--model_size large
```

## ⚙️ Configuration Options

### Model Configuration
```python
# config/custom_config.py
from config.model_config import OriiO1Config, TransformerConfig, DiffusionConfig

custom_config = OriiO1Config(
    transformer=TransformerConfig(
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        vocab_size=50000
    ),
    diffusion=DiffusionConfig(
        image_size=768,
        num_inference_steps=75
    )
)
```

### Training Configuration
```python
# Custom training config
training_config = TrainingConfig(
    batch_size=16,
    learning_rate=3e-4,
    num_epochs=25,
    mixed_precision="bf16",
    gradient_accumulation_steps=2,
    warmup_steps=2000,
    text_loss_weight=1.0,
    image_loss_weight=1.5,
    multimodal_loss_weight=0.8
)
```

## 📈 Performance Optimization

### Memory Optimization
```bash
# Enable gradient checkpointing
--gradient_checkpointing

# Use mixed precision
--mixed_precision fp16

# Reduce batch size if OOM
--batch_size 2 --gradient_accumulation_steps 8
```

### Speed Optimization
```bash
# Use flash attention (if available)
--use_flash_attention

# Optimize for inference
--dtype float16 --device cuda
```

## 🔧 Advanced Usage

### Custom Training Data Processing
```python
from src.orii_o1.training.data_loader import OriiO1Dataset

# Custom dataset
dataset = OriiO1Dataset(
    data_dir="./my_data",
    mode="multimodal",
    max_text_length=1024,
    image_size=768
)
```

### Custom Generation Parameters
```python
# Text generation
response = model.text_model.generate(
    input_ids,
    max_new_tokens=200,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.15,
    do_sample=True
)

# Image generation
image = model.generate_image(
    prompt_embeds,
    height=768,
    width=768,
    num_inference_steps=75,
    guidance_scale=9.0
)
```

### API Integration
```python
import requests

# Text generation API
response = requests.post('http://localhost:5000/api/chat', 
    json={
        'message': 'Hello!',
        'history': []
    }
)

# Image generation API
response = requests.post('http://localhost:5000/api/generate_image',
    json={
        'prompt': 'A beautiful landscape',
        'width': 512,
        'height': 512
    }
)
```

## 🛠️ Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```bash
# Reduce batch size
--batch_size 1 --gradient_accumulation_steps 16

# Use CPU offloading
--offload_to_cpu

# Use smaller model
--model_size small
```

#### Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Use mixed precision
--mixed_precision fp16

# Increase batch size if memory allows
--batch_size 8
```

#### Poor Quality Generation
```bash
# Increase training epochs
--num_epochs 20

# Better learning rate scheduling
--warmup_steps 2000

# Higher quality data
# - More diverse training examples
# - Better image-text pairs
# - Longer conversations
```

### Performance Monitoring
```python
# Monitor training with TensorBoard
tensorboard --logdir ./models/checkpoints/logs

# Monitor with Weights & Biases
# Set use_wandb=True in training config
```

## 📚 Data Preparation Guide

### Text Data Best Practices
1. **Diversity**: Include various conversation styles and topics
2. **Quality**: Clean, well-formatted text without errors
3. **Length**: Mix of short responses and longer explanations
4. **Context**: Include conversation history for better context understanding

### Image Data Best Practices
1. **Resolution**: High-quality images (512x512 minimum)
2. **Captions**: Detailed, descriptive captions
3. **Diversity**: Various subjects, styles, and compositions
4. **Format**: Use standard formats (JPG, PNG)

### Conversation Data Format
```json
{
  "conversation": [
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I don't have access to real-time weather data, but I can help you understand weather patterns or suggest ways to check the weather!"}
  ]
}
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_training.py
pytest tests/test_inference.py
```

### Model Validation
```bash
# Validate model outputs
python scripts/validate_model.py \
    --model_path ./models/checkpoints/best_model.pt \
    --test_data ./data/validation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Transformer architecture inspired by recent advances in language modeling
- Diffusion model based on DDPM and Stable Diffusion research
- Community feedback and contributions

## 📞 Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and community support

## 🔮 Roadmap

- [ ] **v1.1**: Enhanced multimodal fusion
- [ ] **v1.2**: Video generation capabilities
- [ ] **v1.3**: Improved efficiency optimizations
- [ ] **v1.4**: Mobile deployment support
- [ ] **v2.0**: Next-generation architecture

---

**Built with ❤️ for the open-source AI community**
