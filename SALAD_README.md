# 🎯 Guida Rapida Salad Cloud

## 📝 Setup (5 minuti)

### 1. Ottieni Credenziali Kaggle

1. Vai su https://www.kaggle.com/settings
2. Sezione **API** → Clicca **Create New Token**
3. Si scarica `kaggle.json` con:
   ```json
   {
     "username": "tuo_username",
     "key": "abc123..."
   }
   ```

### 2. Modifica Script

Apri `salad_run.sh` e modifica queste righe (riga 14-15):

```bash
KAGGLE_USERNAME="tuo_username"  # Dal kaggle.json
KAGGLE_KEY="abc123..."          # Dal kaggle.json
```

### 3. Push su GitHub

```bash
cd /Users/mariomastrulli/Documents/GitHub/DeepLearningHard_ISWM
git add salad_run.sh
git commit -m "Add Salad training script"
git push
```

---

## 🚀 Deploy su Salad (2 minuti)

### 1. Vai su Salad Portal
https://portal.salad.com/

### 2. Crea Container Group

Clicca **Create Container Group**

**Settings:**
```
Name: crack-detection
Image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

vCPU: 4-8
Memory: 16384 MB (16 GB)
GPU: RTX 3060 / RTX 3070 (12GB VRAM)
Storage: 50 GB
```

**Command:**
```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Biobay/DeepLearningHard_ISWM/main/salad_run.sh)"
```

### 3. Deploy

Clicca **Deploy** → Aspetta che parta (~2 min)

---

## 📊 Monitora Training

1. Vai su **Logs** nel Salad Portal
2. Vedrai il progresso in tempo reale:
   ```
   📦 Cloning repository...
   📥 Downloading dataset from Kaggle...
   🚀 STARTING TRAINING (50 epochs)
   Epoch 1/50 - Train Loss: 0.1234
   ...
   ✅ ALL COMPLETED!
   ```

**Durata totale: ~2.5-3 ore**

---

## 💾 Scarica Risultati

Quando vedi "ALL COMPLETED!" nei logs:

### Metodo A: Dal Portal

1. Container → **Files/Shell**
2. Naviga in `/workspace/DeepLearningHard_ISWM`
3. Scarica:
   - `models/best_autoencoder.pth`
   - `predictions/` (tutte le maschere)
   - `results_visualization.png`
   - `threshold_optimization.png`

### Metodo B: Container Shell

```bash
cd /workspace/DeepLearningHard_ISWM
tar -czf results.tar.gz models/ predictions/ *.png
# Poi scarica results.tar.gz dal portal
```

---

## 💰 Costi

- **RTX 3060**: $0.20/ora
- **Training**: 2.5 ore = $0.50
- **Setup**: 0.2 ore = $0.04
- **TOTALE**: ~**$0.54**

---

## 🎯 Checklist Completa

- [ ] Ottieni `kaggle.json` da Kaggle
- [ ] Modifica `salad_run.sh` con username e key
- [ ] Push su GitHub
- [ ] Crea container su Salad Portal
- [ ] Command: `bash -c "$(curl -fsSL https://raw.githubusercontent.com/Biobay/DeepLearningHard_ISWM/main/salad_run.sh)"`
- [ ] Deploy
- [ ] Monitora logs (2-3 ore)
- [ ] Scarica risultati

---

## 🔧 Troubleshooting

**"401 Unauthorized" da Kaggle**
→ Verifica username/key corretti in `salad_run.sh`

**"Out of Memory"**
→ Modifica `config.py`: `BATCH_SIZE = 16` o `IMAGE_SIZE = (64, 64)`

**Container si ferma**
→ Normale su Salad spot instances. Riavvia: riprende da checkpoint

**Script non trovato (404)**
→ Aspetta 2-3 minuti dopo push, cache GitHub CDN

---

## ✅ Fatto!

Script **completamente automatico**:
- ✅ Clona repo
- ✅ Scarica dataset da Kaggle (2.1 GB)
- ✅ Training 50 epoche
- ✅ Inference + Evaluation
- ✅ Risultati pronti

**Zero upload manuale - tutto automatico!** 🎉
