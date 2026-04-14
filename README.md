# ModL - 3D Model AI Training & Generation System

**Status:** Sprint 1 Complete ✅  
**Target Framework:** .NET 10 / C# 14

## Overview

ModL is an ambitious AI system for training on annotated 3D models and generating new 3D models from text or image prompts using a multi-stage neural network architecture.

## 🚀 Quick Start

### Prerequisites
- .NET 10 SDK
- Visual Studio 2026 Community (optional)
- CUDA 12+ for GPU support (optional)
- 16GB+ RAM recommended

### Build

```bash
dotnet restore
dotnet build
```

### Run Training CLI

```bash
cd ModL.ConsoleApp
dotnet run -- --help
```

### Run Generator CLI

```bash
cd ModL.Generator
dotnet run -- --help
```

## 📁 Project Structure

```
ModL/
├── ModL.Core/              # Core 3D processing library
│   ├── Geometry/           # 3D data structures (Model3D, Mesh, Material)
│   ├── IO/                 # Model loaders/exporters (OBJ, FBX)
│   ├── Voxel/              # Voxelization engine
│   └── Rendering/          # Multi-view renderer
│
├── ModL.ML/                # Machine learning models (Sprint 3-4)
│   └── Models/             # Neural network architectures (TBD)
│
├── ModL.Data/              # Dataset management
│   ├── Datasets/           # Dataset downloaders (ShapeNet, ModelNet)
│   ├── Annotations/        # Annotation parsers
│   └── Pipeline/           # Preprocessing pipeline
│
├── ModL.ConsoleApp/        # Training CLI application
│   └── Commands:
│       ├── download        # Download datasets
│       ├── preprocess      # Prepare data for training
│       ├── train           # Train ML models (TBD)
│       ├── evaluate        # Evaluate models (TBD)
│       └── export          # Export embeddings (TBD)
│
├── ModL.Generator/         # Generation CLI application (Sprint 6)
│   └── Commands:
│       ├── text            # Text-to-3D generation
│       └── image           # Image-to-3D generation
│
├── plan.md                 # Full implementation plan
└── spec.md                 # Technical specification
```

## 🎯 Sprint 1: Foundation (COMPLETED)

### Implemented Features

#### ✅ Project Structure
- 5 projects created with proper dependencies
- Solution file configured
- NuGet packages integrated

#### ✅ Core 3D Processing (ModL.Core)
- **Model3D** - Complete 3D model representation
- **Mesh** - Vertex, index, normal, UV data structures
- **Material** - PBR material system
- **GeometryUtils** - Mesh operations and calculations
- **BoundingBox** - Spatial calculations

#### ✅ 3D File I/O
- **OBJ Loader** - Full Wavefront OBJ/MTL support
- **OBJ Exporter** - Export with materials and textures
- **FBX Loader** - Placeholder for future implementation
- **ModelIOFactory** - Extensible loader/exporter registration

#### ✅ Voxelization
- **VoxelGrid** - 3D voxel grid representation
- **Voxelizer** - Triangle-to-voxel conversion
- **Marching Cubes** - Voxel-to-mesh reconstruction (basic)
- Support for resolutions: 32³, 64³, 128³, 256³, 512³

#### ✅ Multi-View Rendering
- **MultiViewRenderer** - Wireframe rendering from multiple angles
- **ViewConfiguration** - Camera positioning system
- Standard view generation (12-24 discrete angles)
- Orthographic views (front, back, left, right, top, bottom)

#### ✅ Dataset Management (ModL.Data)
- **ShapeNet Downloader** - Instructions for manual download
- **ModelNet Downloader** - Automated HTTP download
- **Annotation Parsers** - JSON annotation support
- **DataProcessor** - Parallel preprocessing pipeline

#### ✅ Preprocessing Pipeline
- Model normalization (centering, scaling to unit cube)
- Automatic normal calculation
- Voxelization at configurable resolutions
- Multi-view rendering (12-24 views)
- Feature extraction (geometric properties)
- Batch processing with progress reporting

#### ✅ Training CLI (ModL.ConsoleApp)
- **download** command - Dataset acquisition
- **preprocess** command - Data preparation
- **train** command - Placeholder for Sprint 5
- **evaluate** command - Placeholder for Sprint 5
- **export** command - Placeholder for Sprint 5
- Rich terminal UI with Spectre.Console
- Progress bars and status reporting
- Structured logging with Serilog

#### ✅ Generator CLI (ModL.Generator)
- Project created with placeholder
- Ready for Sprint 6 implementation

## 📊 Supported Datasets

### ShapeNet
- **Models:** 51,300+ across 55 categories
- **Annotations:** Rich semantic annotations and part segmentations
- **Format:** OBJ, binvox
- **Download:** Manual (requires registration at https://shapenet.org/)

### ModelNet
- **Models:** 127,915 CAD models, 662 categories
- **Format:** OFF (convertible to OBJ)
- **Download:** Automated via CLI

### Future Support
- PartNet (fine-grained annotations)
- ABC Dataset (CAD models)
- Objaverse (1M+ user-generated models)

## 🛠️ CLI Usage Examples

### Download a Dataset

```bash
# ShapeNet (manual download required)
modl download shapenet ./datasets/shapenet

# ModelNet (automated)
modl download modelnet ./datasets/modelnet
```

### Preprocess Models

```bash
# Basic preprocessing
modl preprocess ./datasets/shapenet/raw ./datasets/shapenet/processed

# Advanced preprocessing
modl preprocess ./input ./output \
  --voxel-size 128 \
  --views 24 \
  --parallel 16
```

### Training (Coming in Sprint 5)

```bash
modl train ./configs/default-config.yaml \
  --gpu 0 \
  --resume ./models/checkpoint-epoch-50.pth
```

## 🏗️ Multi-Stage ML Architecture (Sprint 3-4)

### Planned Network Stages

1. **Voxel Encoder** (3D CNN)
   - Input: 64³ voxel grid
   - Output: 512-dim latent vector
   - Purpose: Capture overall shape and topology

2. **Mesh Detail Encoder** (PointNet++)
   - Input: Normalized mesh
   - Output: 512-dim latent vector
   - Purpose: Capture surface details and geometry

3. **Texture Encoder** (2D CNN/ResNet)
   - Input: UV texture maps
   - Output: 256-dim latent vector
   - Purpose: Capture material and color information

4. **Multi-View Encoder** (Multi-view CNN)
   - Input: 12 rendered views (512x512)
   - Output: 512-dim latent vector
   - Purpose: Rotation-invariant representation

5. **Fusion Network** (Transformer)
   - Inputs: All latent vectors from stages 1-4
   - Output: Unified 1024-dim embedding
   - Purpose: Combine all representations

6. **Conditional Generator** (VAE/Diffusion)
   - Inputs: Text/image embedding + unified embedding
   - Outputs: Voxel grid, mesh, texture
   - Purpose: Generate new 3D models

## 🎨 Generation Examples (Sprint 6)

```bash
# Text-to-3D
modl-gen text "a blue sports car" --output car.obj --resolution 512

# Image-to-3D
modl-gen image reference.jpg --output model.fbx --resolution 1024 --texture-size 2048

# Advanced generation
modl-gen text "medieval castle" \
  --output castle.obj \
  --resolution 2048 \
  --temperature 0.8 \
  --texture-size 4096 \
  --format fbx \
  --seed 42
```

## 📦 Dependencies

### Core Libraries
- **SixLabors.ImageSharp** - Image processing
- **SharpGLTF.Toolkit** - glTF/GLB support
- **System.Numerics.Tensors** - Tensor operations

### ML Libraries (Sprint 3+)
- **Microsoft.ML** - ML.NET framework
- **TorchSharp** - Deep learning
- **Microsoft.ML.OnnxRuntime.Gpu** - GPU inference

### CLI & Utilities
- **CommandLineParser** - Argument parsing
- **Spectre.Console** - Rich terminal UI
- **Serilog** - Structured logging
- **YamlDotNet** - Configuration files
- **Newtonsoft.Json** - JSON processing

## ⚠️ Known Issues

1. **ImageSharp Vulnerability Warnings**
   - Version 3.1.6 has known vulnerabilities
   - Acceptable for development/research
   - Update to latest stable before production use

2. **FBX Loader Not Implemented**
   - Placeholder exists
   - Requires AssimpNet integration or custom implementation
   - OBJ format works fully

3. **Visual Studio Build Cache**
   - VS may show false errors due to project reload
   - Use `dotnet build` from command line if issues persist
   - Restart Visual Studio to clear cache

## 🎯 Next Steps (Sprint 2-6)

### Sprint 2: Data Preparation (Weeks 2-3)
- [ ] Test ShapeNet preprocessing pipeline
- [ ] Implement data augmentation
- [ ] Create dataset split utilities
- [ ] Build ML data loaders

### Sprint 3: ML Models - Encoders (Weeks 3-4)
- [ ] Implement voxel encoder (3D CNN)
- [ ] Implement mesh encoder (PointNet++)
- [ ] Implement texture encoder (ResNet)
- [ ] Implement multi-view encoder
- [ ] Create fusion network

### Sprint 4: ML Models - Generator (Week 5)
- [ ] Implement VAE/Diffusion decoder
- [ ] Create mesh generator
- [ ] Create texture generator
- [ ] Implement format exporters

### Sprint 5: Training App (Week 6)
- [ ] Build training loop
- [ ] Add TensorBoard logging
- [ ] Create checkpoint management
- [ ] Build evaluation pipeline

### Sprint 6: Generation App (Weeks 7-8)
- [ ] Integrate CLIP for text encoding
- [ ] Build inference pipeline
- [ ] Implement all CLI parameters
- [ ] Add post-processing

## 📈 Success Metrics

### Training Quality (Sprint 5)
- Reconstruction loss < 0.01
- Classification accuracy > 90%
- IoU (voxels) > 0.85
- Chamfer distance (meshes) < 0.001

### Generation Quality (Sprint 6)
- User preference score > 70%
- CLIP alignment score > 0.25
- Manifold meshes (no holes)
- Seamless UV mapping

### Performance
- Training: < 72 hours on single GPU
- Inference: < 30 seconds per model
- Memory: < 16GB RAM for inference

## 📄 License

This is a research/educational project. Dataset licenses vary:
- **ShapeNet:** Research use, attribution required
- **ModelNet:** Academic use
- **Generated models:** Check your institutional/commercial usage rights

## 🤝 Contributing

This is currently a single-developer project for learning purposes. Contributions, ideas, and feedback are welcome!

## 📚 References

- [ShapeNet](https://shapenet.org/) - 3D model dataset
- [PointNet++](https://arxiv.org/abs/1706.02413) - Point cloud neural networks
- [CLIP](https://openai.com/research/clip) - Text-image embeddings
- [Marching Cubes](https://en.wikipedia.org/wiki/Marching_cubes) - Voxel-to-mesh algorithm

## 📞 Contact

For questions or discussion about this project, please open an issue on GitHub.

---

**Last Updated:** 2025  
**Version:** 0.1.0 (Sprint 1 Complete)  
**Status:** Active Development 🚧
