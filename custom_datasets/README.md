
# Plant Image Dataset

This dataset contains curated plant images downloaded from public sources and filtered using NLP and resolution-based heuristics. Images are organized into folders based on resolution and labeled for use in downstream machine learning tasks such as classification, segmentation, or growth modeling.

---

## 📁 Directory Structure

```

/\[base\_dir]/
├── clean/
│   └── \[label]/
│       └── \[width]/
│           └── \[height]/
│               ├── image.jpg
│               ├── image.json
│
├── rejected/
│   └── \[label]/
│       ├── rejected\_image.jpg
│       ├── rejected\_image.json

```

- **Accepted images** are saved under `clean/` with resolution-based folders for traceability.
- Each image is accompanied by a `.json` metadata file containing:
  - `alt_text`
  - `label`
  - `image URL`
  - `dimensions`
  - `reasons` (if rejected, else `"valid"`)

---

## ✅ Inclusion Criteria

### 📏 Image Resolution

- **Minimum width:** `3860 px`  
- **Minimum height:** `3772 px`  
- This threshold reflects the **75th percentile** of image sizes (~15.3MP), ensuring high-fidelity visual detail for vision models.

### 🧠 Alt Text Filtering

Each image is evaluated using [spaCy](https://spacy.io/) NLP. Images are **excluded** if their alt text contains:

- ❌ **Bad lemmas** (e.g., `"cat"`, `"chair"`, `"macro"`, `"field"`, `"laptop"`)
- ❌ **Bad phrases** (e.g., `"potted plant"`, `"interior design"`, `"power plant"`)
- ❌ **Non-ASCII characters**
- ❌ **Empty or whitespace-only alt text**

**Filtering removes:**

- Human presence, body parts, or professions  
- Animals and insects  
- Indoor decor, office supplies, or furniture  
- Landscape/macro/extreme compositions  
- Industrial, religious, or staged environments

---

## 💡 Goals

This dataset is designed to support:

- **Real-world plant imagery** suitable for mid-range field conditions  
- **Balanced composition** without macro noise or scenic distractions  
- **Standardized preprocessing** for use in 512×512 training pipelines or tile-based segmentation

---

## 📝 Notes

- All metadata is stored as structured `.json` files next to each image.  
- Rejected images are saved in a parallel folder with rejection reasons.  
- No human labeling is used—entire pipeline is automated via NLP + heuristics.

---

## 🔧 Planned Improvements

- Add preprocessing to crop or pad all images to a consistent aspect ratio  
- Generate a global metadata CSV or DuckDB table for easy filtering  
- Integrate perceptual hashing or CLIP embeddings for visual clustering  
- Optional user feedback loop for manual validation
```
