# PyTorch Training

**Hands-on training for PyTorch deep learning fundamentals**

## Quick Start

### Option 1: View Documentation as Website (Recommended)

The documentation is built with **VitePress** and can be viewed as an interactive website:

```bash
# 1. Install dependencies
npm install

# 2. Start the documentation server
npm run docs:dev

# 3. Open in browser
# Documentation will be available at http://localhost:5173
```

**Features**:
- 📖 Clean, readable interface
- 🔍 Built-in search functionality
- 📱 Mobile-responsive design
- 🎨 Syntax highlighting for code blocks
- 🔄 Hot-reload during development

### Option 2: Read Markdown Directly

Browse the [Study Guide](./docs/) in the `docs/` folder directly:

```
Start here:  docs/README.md  →  Complete study guide
Then:        module-01/      →  Hands-on labs
```

## Course Overview

| Module | Topic | Description | Technologies |
|--------|-------|-------------|--------------|
| 1 | PyTorch Fundamentals | Tensors, operations, and tensor operations | PyTorch, NumPy |
| 2 | PyTorch Workflow Fundamentals | Data handling, model building, training loop | PyTorch, torch.nn, torch.optim |
| 3 | Neural Network Classification | Classification models, evaluation, and deployment | PyTorch, torchvision, sklearn |

## Repository Structure

```
ml_pytorch_training/
├── docs/                          # 📖 CONCEPTUAL LEARNING
│   ├── README.md                  #   Study guide and navigation
│   ├── module-01/                 #   Module 1: PyTorch Fundamentals
│   ├── module-02/                 #   Module 2: PyTorch Workflow Fundamentals
│   └── module-03/                 #   Module 3: Neural Network Classification
│
├── module-01/                     # 🛠️ LAB & PRACTICE CODE
│   └── pytorch-fundamentals/      #   Tensors and operations
│
├── module-02/                     #   PyTorch Workflow labs
│   └── pytorch-workflow/          #   Data loading and training loops
│
├── module-03/                     #   Neural Network Classification labs
│   └── neural-network-classification/  #   Classification models
│
└── assets/                        # Images and diagrams
```

## How to Use This Training

### For Each Topic

1. **Read the theory** in `docs/module-X/`
2. **Practice with labs** in `module-X/`
3. **Experiment** with code
4. **Build your own** variations

### Example: Learning PyTorch Fundamentals

```bash
# 1. Read the conceptual guide
cat docs/module-01/tensor-basics.md

# 2. Navigate to the lab
cd module-01/pytorch-fundamentals

# 3. Run the exercises
python 01_tensor_creation.py

# 4. Experiment and learn
python 02_tensor_operations.py
```

### Example: Learning Neural Network Classification

```bash
# 1. Read the conceptual guide
cat docs/module-03/classification-basics.md

# 2. Navigate to the lab
cd module-03/neural-network-classification

# 3. Run the training script
python train_classifier.py

# 4. Evaluate the model
python evaluate_model.py
```

## Viewing Documentation with VitePress

This project uses **VitePress** to provide a beautiful, searchable documentation website.

### Installation

```bash
# Install Node.js dependencies
npm install
```

**Requirements**:
- Node.js 18.x or higher
- npm 9.x or higher

### Development Server

Start the local development server with hot-reload:

```bash
npm run docs:dev
```

Open your browser to: **http://localhost:5173**

The development server supports:
- 🔄 **Hot reload**: Changes to markdown files are instantly reflected
- 🔍 **Full search**: Search across all documentation
- 📱 **Responsive**: Works on desktop, tablet, and mobile

### Build for Production

Create a static site ready for deployment:

```bash
# Build the static site
npm run docs:build

# Preview the built site
npm run docs:preview
```

The built site will be in `docs/.vitepress/dist/` and can be deployed to:
- GitHub Pages
- Netlify
- Vercel
- Any static hosting service

### VitePress Commands

| Command | Description |
|---------|-------------|
| `npm install` | Install VitePress dependencies |
| `npm run docs:dev` | Start development server at http://localhost:5173 |
| `npm run docs:build` | Build static site for production |
| `npm run docs:preview` | Preview the production build locally |

### Customization

VitePress configuration is in `docs/.vitepress/config.ts`:
- Navigation menu
- Sidebar structure
- Theme settings
- Search configuration

## Module Guides

| Module | Study Guide | Lab Location |
|--------|-------------|--------------|
| **Module 1** | [PyTorch Fundamentals](./docs/module-01/) | [`module-01/`](./module-01/) |
| **Module 2** | [PyTorch Workflow](./docs/module-02/) | [`module-02/`](./module-02/) |
| **Module 3** | [Neural Network Classification](./docs/module-03/) | [`module-03/`](./module-03/) |

## Prerequisites

- **Node.js** 18.x or higher (for VitePress documentation viewer)
- **npm** 9.x or higher
- **Python** 3.8 or higher
- Basic Python knowledge
- Understanding of machine learning concepts
- Command-line interface familiarity
- Basic linear algebra knowledge

## Study Path

1. **[Start with the Study Guide](./docs/)** - Complete overview
2. **Module 1: PyTorch Fundamentals** - Tensors, operations, and tensor manipulation
3. **Module 2: PyTorch Workflow** - Data handling, model building, and training loops
4. **Module 3: Neural Network Classification** - Classification models, evaluation, and deployment

## Contributing

This is a training repository. See [CONTRIBUTING.md](./CONTRIBUTING.md)

---

## Quick Reference

```bash
# View documentation as website
npm install
npm run docs:dev
# Open http://localhost:5173

# Build for deployment
npm run docs:build
```

**Start Learning:**
- 🌐 **View as Website**: `npm run docs:dev` then open http://localhost:5173
- 📖 **Read as Markdown**: [docs/README.md](./docs/) ← Complete study guide
