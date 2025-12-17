# 🚀 Setup Automatico Salad con Kaggle Dataset

## 📋 Passo 1: Ottieni Credenziali Kaggle

### 1.1 Crea Account Kaggle (se non ce l'hai)
- Vai su https://www.kaggle.com
- Registrati gratuitamente

### 1.2 Genera API Token
1. Vai su https://www.kaggle.com/settings
2. Scorri fino a **API** section
3. Clicca **Create New Token**
4. Verrà scaricato `kaggle.json` con questo contenuto:
   ```json
   {
     "username": "tuo_username",
     "key": "abc123def456..."
   }
   ```

### 1.3 Copia Username e Key
Apri il file `kaggle.json` e copia:
- `username` → esempio: `mariomastrulli`
- `key` → esempio: `f4a3b2c1d0e9f8g7h6i5j4k3l2m1n0o1`

## 📝 Passo 2: Modifica Script (sul tuo Mac)

Apri `/Users/mariomastrulli/Documents/GitHub/DeepLearningHard_ISWM/auto_salad.py`

Trova queste righe e sostituisci con i tuoi dati:

```python
KAGGLE_USERNAME = "mariomastrulli"  # ← Il tuo username Kaggle
KAGGLE_KEY = "f4a3b2c1d0e9..."     # ← La tua API key Kaggle
```

## 🔄 Passo 3: Push su GitHub

```bash
cd /Users/mariomastrulli/Documents/GitHub/DeepLearningHard_ISWM

git add auto_salad.py
git commit -m "Add Kaggle dataset download"
git push origin main
```

## 🎮 Passo 4: Crea Container su Salad

### Vai su: https://portal.salad.com/

**Create Container Group:**

```yaml
Name: crack-detection-kaggle

Image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

Resources:
  vCPU: 4-8
  Memory: 16384 MB (16 GB)
  GPU: RTX 3060 Ti / RTX 3070 (12GB VRAM)
  Storage: 50 GB
```

**Command:**
```bash
python3 -c "$(curl -fsSL https://raw.githubusercontent.com/Biobay/DeepLearningHard_ISWM/main/auto_salad.py)"
```

**Deploy!** ✅

## 📊 Cosa Succede Automaticamente:

```
1. ⚙️  Installa dipendenze sistema
2. 📦 Clona GitHub repository
3. 🔑 Configura credenziali Kaggle
4. 📥 Scarica dataset da Kaggle (2.1 GB)
5. 📦 Estrae e organizza dataset
6. 🐍 Installa requirements Python
7. 🎮 Verifica GPU
8. 🚀 TRAINING (50 epoche, ~2-3 ore)
9. 🔮 INFERENCE (genera maschere)
10. 📊 EVALUATION (metriche IoU, Dice)
11. ⏰ Resta vivo 15 min per download
```

## 💾 Passo 5: Scarica Risultati

Dopo training (vedi logs "TUTTO COMPLETATO!"):

### Shell nel Container:
```bash
cd /workspace/DeepLearningHard_ISWM
ls -lh models/best_autoencoder.pth
ls -lh predictions/
```

### Files da scaricare:
- `models/best_autoencoder.pth` - Modello addestrato
- `predictions/*.jpg` - Maschere predette
- `results_visualization.png` - Visualizzazioni
- `threshold_optimization.png` - Ottimizzazione threshold

## 💰 Costi Stimati

- **Setup + Download**: ~5 min = $0.02
- **Training**: ~2.5 ore = $0.50
- **Inference + Eval**: ~15 min = $0.05
- **TOTALE**: **~$0.57** con RTX 3060

## 🔒 Sicurezza Credenziali

### ⚠️ IMPORTANTE:
Le tue credenziali Kaggle sono nel file `auto_salad.py` su GitHub pubblico!

**Opzione A - Repository Privato (Consigliato):**
```bash
# Rendi repository privato su GitHub:
# Settings → Danger Zone → Change visibility → Make private
```

**Opzione B - Usa Environment Variables su Salad:**

Invece di mettere credenziali nel file, usa Salad Environment Variables:

Nel Salad Portal, aggiungi:
```
KAGGLE_USERNAME=mariomastrulli
KAGGLE_KEY=f4a3b2c1d0e9...
```

Poi modifica `auto_salad.py`:
```python
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "YOUR_KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "YOUR_KAGGLE_API_KEY")
```

## 🔧 Troubleshooting

### "401 Unauthorized" da Kaggle
- Verifica username e key siano corretti
- Ricontrolla `kaggle.json` scaricato

### "Dataset non trovato"
- Verifica link dataset: `lakshaymiddha/crack-segmentation-dataset`
- Controlla su Kaggle che dataset sia pubblico

### "Permission denied kaggle.json"
- Script imposta automaticamente chmod 600
- Se fallisce, è problema permessi container

## 🎯 TL;DR - Quick Commands

```bash
# 1. Ottieni credenziali da https://www.kaggle.com/settings
# 2. Modifica auto_salad.py con username e key
# 3. Push su GitHub
git add auto_salad.py && git commit -m "Add Kaggle credentials" && git push

# 4. Su Salad Portal:
# - Image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# - GPU: RTX 3060, 16GB RAM
# - Command: python3 -c "$(curl -fsSL https://raw.githubusercontent.com/Biobay/DeepLearningHard_ISWM/main/auto_salad.py)"

# 5. Deploy e aspetta ~3 ore
```

✅ Zero upload manuale - tutto automatico da Kaggle!
