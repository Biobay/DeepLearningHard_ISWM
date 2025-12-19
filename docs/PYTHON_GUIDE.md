# 🎯 Guida Completa - File Python Singolo

## ✅ File Creato: `salad_complete.py`

Un **singolo file Python** che fa tutto automaticamente!

---

## 📝 Setup (2 minuti)

### 1. Ottieni Credenziali Kaggle

1. Vai su https://www.kaggle.com/settings
2. Sezione **API** → **Create New Token**
3. Si scarica `kaggle.json`:
   ```json
   {
     "username": "tuo_username",
     "key": "abc123def456..."
   }
   ```

### 2. Modifica `salad_complete.py`

Apri il file e modifica righe 16-17:

```python
KAGGLE_USERNAME = "tuo_username"  # Dal kaggle.json
KAGGLE_KEY = "abc123def456..."    # Dal kaggle.json
```

### 3. Push su GitHub

```bash
git add salad_complete.py
git commit -m "Add complete Python training script"
git push
```

---

## 🚀 Deploy su Salad

### Su Salad Portal: https://portal.salad.com/

**Create Container Group:**

```yaml
Name: crack-detection
Image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

vCPU: 4-8
Memory: 16384 MB (16 GB)
GPU: RTX 3060 / RTX 3070 (12GB VRAM)
Storage: 50 GB
```

**Command (copia tutto):**
```bash
python3 -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/Biobay/DeepLearningHard_ISWM/main/salad_complete.py').read())"
```

**Deploy!** ✅

---

## 📊 Cosa Fa Automaticamente

```
1.  📦 Installa git, wget, curl, unzip
2.  📥 Clona repository GitHub
3.  🔑 Configura credenziali Kaggle
4.  🐍 Installa requirements.txt
5.  🎮 Verifica GPU disponibile
6.  📥 Scarica dataset da Kaggle (2.1 GB)
7.  📦 Estrae e organizza dataset
8.  📁 Crea directory (models, predictions, etc.)
9.  🚀 TRAINING (50 epoche, ~2.5 ore)
10. 🔮 INFERENCE (genera maschere)
11. 📊 EVALUATION (IoU, Dice, F1-score)
12. ⏰ Resta vivo 30 min per scaricare risultati
```

**Durata totale: ~3 ore**

---

## 💾 Scarica Risultati

Quando vedi "ALL COMPLETED!" nei logs:

### Dal Salad Portal:

1. Container → **Files**
2. Naviga: `/workspace/DeepLearningHard_ISWM`
3. Scarica:
   - `models/best_autoencoder.pth`
   - `predictions/` (tutte le maschere)
   - `*.png` (visualizzazioni)

---

## 💰 Costi

- **RTX 3060**: $0.20/ora
- **Totale 3 ore**: **~$0.60**

---

## 🎯 Checklist

- [ ] Ottieni `kaggle.json` da Kaggle
- [ ] Modifica `salad_complete.py` con username/key
- [ ] Push su GitHub
- [ ] Su Salad: Deploy con comando Python
- [ ] Monitora logs (3 ore)
- [ ] Scarica risultati

---

## 🔧 Troubleshooting

**"401 Unauthorized" Kaggle**
→ Verifica username/key corretti

**"ModuleNotFoundError"**
→ Lo script installa tutto automaticamente, aspetta

**Container si ferma**
→ Riavvia, riprende da checkpoint

---

## ✅ Vantaggi File Python

- ✅ **Un singolo file** - tutto in uno
- ✅ **Esecuzione diretta** via curl
- ✅ **Log colorati** con timestamp
- ✅ **Error handling** completo
- ✅ **Zero configurazione manuale**

**È letteralmente un comando e aspetti 3 ore!** 🚀
