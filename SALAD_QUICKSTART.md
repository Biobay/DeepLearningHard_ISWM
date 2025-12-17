# 🚀 Guida Rapida: Setup Automatico su Salad

## 📋 Passo 1: Prepara Dataset (sul tuo Mac)

```bash
# Vai nella cartella progetto
cd /Users/mariomastrulli/Documents/GitHub/DeepLearningHard_ISWM

# Comprimi dataset
tar -czf dataset.tar.gz dataset/

# Controlla dimensione
ls -lh dataset.tar.gz
```

## ☁️ Passo 2: Carica Dataset su Google Drive

1. Vai su https://drive.google.com
2. Upload `dataset.tar.gz`
3. Tasto destro sul file → **Share** → **Anyone with the link**
4. Copia il link (esempio: `https://drive.google.com/file/d/1ABC_XYZ123/view`)
5. Converti in link diretto:
   ```
   Link originale: https://drive.google.com/file/d/1ABC_XYZ123/view
   Link diretto:   https://drive.google.com/uc?export=download&id=1ABC_XYZ123
   ```

## 📝 Passo 3: Modifica Script (sul tuo Mac)

Apri `auto_salad.py` e modifica questa riga:

```python
DATASET_URL = "https://drive.google.com/uc?export=download&id=TUO_FILE_ID"
```

Sostituisci `TUO_FILE_ID` con l'ID del tuo file!

## 🔄 Passo 4: Push su GitHub

```bash
cd /Users/mariomastrulli/Documents/GitHub/DeepLearningHard_ISWM

git add auto_salad.py
git commit -m "Add auto salad script with dataset URL"
git push origin main
```

## 🎮 Passo 5: Crea Container su Salad

### Vai su: https://portal.salad.com/

**Create Container Group:**

```yaml
Name: crack-detection-auto

Image Source: Docker Hub
Image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

Resources:
  vCPU: 4-8
  Memory: 16384 MB (16 GB)
  GPU Type: RTX 3060 / RTX 3070 (12GB VRAM)
  Storage: 50 GB

Replica Count: 1

Container Gateway: Disabled (non serve)
```

**Command Type:** Shell

**Command:**
```bash
python3 -c "$(curl -fsSL https://raw.githubusercontent.com/Biobay/DeepLearningHard_ISWM/main/auto_salad.py)"
```

**Environment Variables:** (nessuna)

## ✅ Passo 6: Avvia!

1. Clicca **Deploy**
2. Aspetta che il container parta (~2 min)
3. Vai su **Logs** per vedere il progresso in tempo reale

## 📊 Cosa Succede Automaticamente:

```
1. ⚙️  Installa git, wget, curl
2. 📦 Clona repository GitHub
3. 📥 Scarica dataset da Google Drive
4. 📦 Estrae dataset.tar.gz
5. 🐍 Installa requirements.txt
6. 🎮 Verifica GPU disponibile
7. 📁 Crea directory (models, predictions, etc.)
8. 🚀 TRAINING (50 epoche, ~2-3 ore)
9. 🔮 INFERENCE (genera maschere)
10. 📊 EVALUATION (calcola metriche)
11. ⏰ Rimane vivo 15 min per download risultati
```

## 💾 Passo 7: Scarica Risultati

Dopo che training finisce (vedi nei logs "TUTTO COMPLETATO!"):

### Metodo A: Dal Portal Salad

1. Vai su **Container** → **Shell**
2. Esegui:
```bash
cd /workspace/DeepLearningHard_ISWM
ls -lh models/
ls -lh predictions/
```
3. Usa funzione **Download** del portal

### Metodo B: Copia da Logs (per piccoli file)

Il modello best è salvato in:
```
/workspace/DeepLearningHard_ISWM/models/best_autoencoder.pth
```

### Metodo C: Push su GitHub (automatico)

Decommentando questa riga in `auto_salad.py`:
```python
upload_results_to_github(project_path)
```

Poi sul tuo Mac:
```bash
git pull
```

## 💰 Costi

- **GPU RTX 3060**: ~$0.20/ora
- **Training completo**: ~2.5 ore = **~$0.50**
- **+ Overhead setup**: ~10 min = **~$0.03**
- **TOTALE**: **~$0.53** per run completo

## 🔧 Troubleshooting

### "Dataset download fallito"
- Verifica link Google Drive sia pubblico
- Controlla formato link diretto corretto

### "Out of Memory"
- Riduci `BATCH_SIZE` in `config.py` a 16
- Oppure `IMAGE_SIZE` a (64, 64)

### "Container si ferma durante training"
- Normale su Salad (spot instances)
- Riavvia container, riprenderà da checkpoint grazie a `--resume`

## 📝 Note Importanti

✅ **Dataset rimane su Google Drive** - Salad lo scarica solo quando serve
✅ **Checkpoints automatici** ogni 5 epoche
✅ **Resume automatico** se container si interrompe
✅ **Container self-contained** - zero configurazione manuale

---

## 🎯 TL;DR - Comandi Rapidi

```bash
# Sul Mac
cd /Users/mariomastrulli/Documents/GitHub/DeepLearningHard_ISWM
tar -czf dataset.tar.gz dataset/
# Upload su Drive, ottieni link

# Modifica auto_salad.py con link dataset
git add auto_salad.py
git push

# Su Salad Portal:
# Image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# Command: python3 -c "$(curl -fsSL https://raw.githubusercontent.com/Biobay/DeepLearningHard_ISWM/main/auto_salad.py)"
# GPU: RTX 3060, 16GB RAM

# Deploy → Aspetta 2-3 ore → Scarica risultati
```

Fatto! 🎉
