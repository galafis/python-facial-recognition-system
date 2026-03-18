# Facial Recognition System

[Portugues](#portugues) | [English](#english)

---

## English

### Overview

A facial recognition system built from scratch using OpenCV, dlib, and deep learning embeddings. Implements face detection, encoding extraction, and identity matching with a modular pipeline architecture.

**DIO Lab Project** - Formacao Machine Learning Specialist

### Features

- **Face Detection**: Haar cascades, HOG + SVM, CNN-based detection
- **Face Encoding**: 128-dimensional embedding extraction
- **Identity Matching**: Euclidean distance and cosine similarity
- **Database Management**: Face encoding storage and retrieval
- **Real-time Processing**: Video stream face recognition
- **Batch Processing**: Process multiple images at once

### Tech Stack

- Python 3.10+
- OpenCV
- dlib / face_recognition
- NumPy / Pandas
- Docker
- GitHub Actions CI/CD

### Project Structure

```
python-facial-recognition-system/
|-- src/
|   |-- __init__.py
|   |-- face_detector.py
|   |-- face_encoder.py
|   |-- face_matcher.py
|-- tests/
|   |-- __init__.py
|   |-- test_detector.py
|-- .github/
|   |-- workflows/
|       |-- ci.yml
|-- Dockerfile
|-- requirements.txt
|-- README.md
|-- LICENSE
```

### Quick Start

```bash
git clone https://github.com/galafis/python-facial-recognition-system.git
cd python-facial-recognition-system
pip install -r requirements.txt
python -m src.face_detector
```

### Docker

```bash
docker build -t facial-recognition .
docker run --rm facial-recognition
```

### License

MIT License - see [LICENSE](LICENSE).

---

## Portugues

### Visao Geral

Sistema de reconhecimento facial construido do zero usando OpenCV, dlib e embeddings de deep learning. Implementa deteccao facial, extracao de encodings e correspondencia de identidade.

**Projeto Lab DIO** - Formacao Machine Learning Specialist

### Funcionalidades

- **Deteccao Facial**: Haar cascades, HOG + SVM, deteccao baseada em CNN
- **Encoding Facial**: Extracao de embeddings de 128 dimensoes
- **Correspondencia**: Distancia Euclidiana e similaridade cosseno
- **Banco de Dados**: Armazenamento e recuperacao de encodings
- **Processamento em Tempo Real**: Reconhecimento em stream de video
- **Processamento em Lote**: Processar multiplas imagens

### Inicio Rapido

```bash
git clone https://github.com/galafis/python-facial-recognition-system.git
cd python-facial-recognition-system
pip install -r requirements.txt
python -m src.face_detector
```

### Licenca

Licenca MIT - veja [LICENSE](LICENSE).
