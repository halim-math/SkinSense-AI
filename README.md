# SkinSenseAI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

> **A professional dermatology image and education platform powered by uncertainty-aware deep learning**

SkinSenseAI is an **educational** AI-powered platform that provides skin condition risk assessment, triage suggestions, evidence-based care guidance and continuous automatic self-improving. It uses computer vision, uncertainty estimation, and active learning to improve over time—designed as a **research showcase project** demonstrating production-ready ML engineering.

![SkinSenseAI Screenshot](https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1200&h=630&fit=crop)

---

## 🎯 What This Is (And Isn't)

### ✅ What It IS
- **Educational SkinSense tool** providing risk/urgency categories
- **Uncertainty-aware AI** that asks follow-up questions when confidence is low
- **Safety-first design** with built-in rails for severe symptoms
- **Active learning system** improving from user feedback
- **GitHub portfolio showcase** of ML engineering best practices

### ❌ What It Is NOT
- **Not a medical diagnosis tool**
- **Not a replacement for professional healthcare**
- **Not intended for emergency medical decisions**

> Always consult a qualified healthcare provider for medical advice, diagnosis, and treatment.

---

## 🌟 Key Features

### 🔬 Intelligent Analysis
- **Multi-task deep learning model** (condition classification + severity + injury detection)
- **Uncertainty quantification** using MC Dropout / Deep Ensembles
- **Smart follow-up questions** when confidence is low
- **Evidence-based recommendations** with confidence scores

### 🎯 Triage Levels
- **Emergency**: Immediate medical attention required
- **Urgent**: See doctor within 24-48 hours
- **Routine**: Schedule appointment in coming weeks
- **Self-Care**: Monitor at home with care tips

### 🔄 Active Learning Loop
- Feedback collection from users
- Doctor-confirmed labels (optional)
- Prioritized retraining on uncertain cases
- Model versioning and A/B testing

### 🛡️ Safety & Privacy
- HIPAA-compliant data handling
- De-identified image storage
- Safety rails for severe symptoms
- Transparent uncertainty reporting

---

## 🏗️ Architecture

```
SkinSense-ai/
├── frontend/           # React + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── components/   # Reusable UI components
│   │   ├── pages/        # Main application pages
│   │   ├── lib/          # Utilities and mock data
│   │   └── types/        # TypeScript definitions
│   └── public/
│
├── backend/            # FastAPI + PostgreSQL (to be implemented)
│   ├── app/
│   │   ├── api/          # REST API routes
│   │   ├── services/     # Business logic
│   │   ├── db/           # Database models
│   │   └── security/     # Auth & rate limiting
│   └── tests/
│
├── ml/                 # Machine Learning Pipeline (to be implemented)
│   ├── datasets/         # Data versioning (DVC)
│   ├── training/         # Training scripts
│   ├── inference/        # Deployment-ready inference
│   ├── evaluation/       # Metrics & calibration
│   └── model_card.md     # Model documentation
│
└── docs/
    ├── architecture.md
    ├── privacy.md
    └── safety.md
```

---

## 🚀 Tech Stack

### Frontend (Implemented ✅)
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Vite** for fast builds
- **shadcn/ui** component library
- **Lucide React** icons
- Fully responsive (mobile-first)

### Backend (Planned 🔨)
```python
# Core Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9

# Async & Caching
redis==5.0.1
celery==5.3.4

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
```

### ML Pipeline (Planned 🔨)
```python
# Deep Learning - PyTorch
torch==2.1.0
torchvision==0.16.0
timm==0.9.12              # SOTA model architectures

# Computer Vision
opencv-python==4.8.1.78
Pillow==10.1.0
albumentations==1.3.1     # Data augmentation

# ML Utilities
scikit-learn==1.3.2
numpy==1.26.2
pandas==2.1.3

# Uncertainty & Explainability
captum==0.6.0             # Model interpretability
torchmetrics==1.2.0

# MLOps
mlflow==2.8.1             # Experiment tracking
dvc==3.32.0               # Data versioning (optional)

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

---

## 📋 Roadmap

### Phase 1: MVP Frontend ✅ (Current)
- [x] Responsive UI design
- [x] Photo upload with preview
- [x] Mock AI analysis flow
- [x] Triage level display
- [x] Care tips & recommendations
- [x] Feedback mechanism

### Phase 2: Backend API 🔨
- [ ] FastAPI REST endpoints
- [ ] PostgreSQL database setup
- [ ] User authentication (JWT)
- [ ] Image storage (S3/MinIO)
- [ ] Rate limiting & security
- [ ] Async job queue (Celery)

### Phase 3: ML Pipeline 🔨
- [ ] Dataset collection & preprocessing
- [ ] Model training (EfficientNet/ViT baseline)
- [ ] Uncertainty quantification implementation
- [ ] Model calibration (temperature scaling)
- [ ] Inference optimization
- [ ] MLflow experiment tracking

### Phase 4: Advanced Features 🎯
- [ ] Follow-up question system (dynamic)
- [ ] Active learning loop
- [ ] Model retraining pipeline
- [ ] A/B testing framework
- [ ] Admin dashboard (case review)
- [ ] Knowledge base (500+ conditions/medicines)

### Phase 5: Production 🚀
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Monitoring & logging (Prometheus/Grafana)
- [ ] Load testing
- [ ] Security audit
- [ ] HIPAA compliance verification

---

## 🧪 Model Design

### Architecture
```python
class DermaTriage(nn.Module):
    """
    Multi-task dermatology triage model
    """
    def __init__(self):
        # Backbone: EfficientNet-B4 / ConvNeXt / ViT
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True)
        
        # Task heads
        self.condition_head = nn.Linear(1792, num_conditions)    # Multi-class
        self.severity_head = nn.Linear(1792, 4)                   # Triage level
        self.injury_head = nn.Linear(1792, 1)                     # Binary
        
    def forward(self, x):
        features = self.backbone(x)
        
        # Outputs
        condition = self.condition_head(features)
        severity = self.severity_head(features)
        injury = torch.sigmoid(self.injury_head(features))
        
        return condition, severity, injury
```

### Uncertainty Estimation
- **MC Dropout**: Multiple forward passes with dropout enabled
- **Deep Ensembles**: Train 3-5 models, average predictions
- **Temperature Scaling**: Post-hoc calibration for confidence scores

### Training Strategy
1. **Pre-training**: ImageNet weights
2. **Fine-tuning**: Dermatology datasets (HAM10000, Dermnet, etc.)
3. **Active Learning**: Prioritize uncertain cases for labeling
4. **Evaluation**: Balanced accuracy, calibration error, F1-score per class

---

## 📊 Dataset Strategy

### Public Sources
- **HAM10000**: 10,000 dermatoscopic images (7 conditions)
- **Fitzpatrick17k**: Diverse skin tones dataset
- **DermNet**: Educational dermatology images
- **ISIC Archive**: International Skin Imaging Collaboration

### Data Handling
- De-identification of all images
- Balanced sampling across skin tones
- Augmentation: rotation, flip, color jitter, cutout
- Train/val/test split: 70/15/15

### Privacy & Ethics
- HIPAA-compliant storage
- Consent-based data collection
- Bias monitoring across demographics
- Regular fairness audits

---

## 🔒 Safety Rails

### Built-in Warnings
The system automatically escalates to "Emergency" triage if it detects:
- Open wounds with bleeding
- Rapidly spreading rashes
- Severe swelling or blistering
- Signs of infection (fever mentioned)
- Symptoms of allergic reaction

### Disclaimers
- Always shown before analysis
- Repeated in results
- Clear "not a diagnosis" messaging
- Links to emergency services

---

## 🛠️ Development Setup

### Frontend (Current)
```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

### Backend (Coming Soon)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload
```

### ML Pipeline (Coming Soon)
```bash
# Install ML dependencies
pip install -r ml/requirements.txt

# Train model
python ml/training/train.py --config configs/efficientnet_b4.yaml

# Run evaluation
python ml/evaluation/evaluate.py --model-path models/best.pth
```

---

## 📖 Documentation

- **[Model Card](docs/MODEL_CARD.md)**: Intended use, limitations, bias analysis
- **[Data Sheet](docs/DATASHEET.md)**: Dataset details, consent, distribution
- **[Safety Policy](docs/SAFETY.md)**: Triage rules, escalation criteria
- **[Privacy Policy](docs/PRIVACY.md)**: Data handling, retention, compliance
- **[Architecture](docs/ARCHITECTURE.md)**: System design, API specs

---

## 🤝 Contributing

This is a research/showcase project. Contributions are welcome for:
- Bug fixes
- Documentation improvements
- New dataset integrations
- Model architecture experiments

Please open an issue before starting major work.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## ⚠️ Medical Disclaimer

**This software is for educational and research purposes only.**

- **NOT** a medical device
- **NOT** FDA-approved
- **NOT** a substitute for professional medical advice
- **Always** consult a qualified healthcare provider for medical decisions

The creators and contributors assume **no liability** for any medical decisions made using this tool.

---

## 📧 Contact

**Project Maintainer**: [Abdul Halim]
- GitHub: [@halim-math](https://github.com/halim-math)
- Email: abdul.halim@uni-goettingen.de

**For Research Inquiries**: abdul.halim@uni-goettingen.de

---

## 🙏 Acknowledgments

- **Datasets**: HAM10000, Fitzpatrick17k, DermNet, ISIC
- **Frameworks**: PyTorch, FastAPI, React
- **Inspiration**: Real-world dermatology triage challenges

---

## 📈 Project Status

**Current Version**: 1.0.0 (Frontend MVP)
**Status**: 🟢 Active Development
**Next Milestone**: Backend API Implementation

---

**Built with ❤️ for better healthcare through AI**
