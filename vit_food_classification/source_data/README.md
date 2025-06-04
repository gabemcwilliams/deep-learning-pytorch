
# 🌿 Pexels Plant Image Collector & Filter Pipeline

This project is a full data ingestion and filtering pipeline for collecting **plant images** from the Pexels API. It includes NLP-powered filtering using `spaCy` to reject noisy, irrelevant, or mislabeled results, and saves all images with structured metadata.

---

## 🚀 Overview

- 🔍 **API Querying** — Downloads large batches of images from Pexels based on custom search terms.
- 🧠 **Alt-Text Filtering** — Uses spaCy NLP to reject images based on lemma/token/phrase analysis.
- 🗃 **Structured Storage** — Saves accepted images by resolution; rejected images separately with metadata.
- 📦 **Metadata Logging** — Each image has a corresponding `.json` file for reproducibility and later analysis.
- 🧪 **Automation Ready** — Batch processing with support for multi-query lists and resumable Parquet caching.

---

## 📁 Directory Structure

```

/mnt/mls/
├── images/
│   ├── clean/
│   │   └── \[query\_label]/\[width]/\[height]/image.jpg + image.json
│   └── rejected/
│       └── \[query\_label]/image.jpg + image.json
├── data/
│   └── pexels\_metadata/
│       ├── leaf\_veins\_macro.parquet
│       └── ...

````

---

## ✅ Filtering Criteria

### 🧠 Alt Text NLP Filtering

Images are rejected if their alt text contains:

- ❌ **Bad phrases** (e.g., `"potted plant"`, `"interior design"`, `"power plant"`)
- ❌ **Bad lemmas/tokens** (e.g., `"cat"`, `"chair"`, `"macro"`, `"field"`)
- ❌ **Non-ASCII characters** or **empty/blank descriptions**

This removes:
- People, body parts, professions
- Animals and insects
- Indoor furniture and decor
- Industrial objects or scenes
- Scenic, macro, or irrelevant landscape content

Filtering is performed with [spaCy](https://spacy.io/) using the `en_core_web_sm` model.

---

## 🖼 Image Organization

### Accepted Images
- Saved under `clean/` in folders based on `label`, `width`, and `height`
- Each image includes:
  - `.jpg` file
  - `.json` metadata file with `alt_text`, `dimensions`, `reasons`, `filename`, and `URL`

### Rejected Images
- Saved under `rejected/` by label
- Also include `.json` metadata with rejection reasons

---

## 🔧 Requirements

- Python 3.10+
- Dependencies:  
  `spaCy`, `requests`, `pandas`, `pyarrow`, `loguru`, `colorama`, `PIL`, `certifi`, `vault_mgr` (custom)

Install spaCy model:
```bash
python -m spacy download en_core_web_sm
````

---

## 🛠 Usage

1. **Set up Vault secrets** for Pexels API (`api_key`, `base_uri`)
2. **Customize your `query_list`** to define image search topics
3. **Run the script** to download and process images
4. Review saved folders in `/mnt/mls/images` and `.parquet` metadata under `/mnt/mls/data/pexels_metadata`

---

## 📈 Output Summary

The script prints a summary:

```
Clean images: 324
Rejected images: 211
Acceptance rate: 60.55%
```

---

## 🗂 Planned Improvements

* Optional cropping/padding for 512×512 standardized training tiles
* Automatic CLIP embedding + image clustering
* GUI interface for reviewing rejections
* Add support for other datasets (Unsplash, Flickr)

---

## 🔒 Security Notes

* API keys are securely loaded via `VaultManager` and never stored in code
* Directory paths assume local dev environment and are safe for internal use

---

## 👨‍💻 Author Notes

Built as part of a pipeline to create clean, filtered, and resolution-consistent datasets for edge vision tasks involving plant growth, classification, and embedded deployment.

```