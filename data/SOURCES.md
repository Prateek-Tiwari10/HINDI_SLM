# Data sources

Verified 2026-10-01 against Hugging Face metadata (licence from the dataset card; sizes from the file listing and the datasets-server).

| Source | Link | Licence | Subset used | Why |
| --- | --- | --- | --- | --- |
| Sangraha Verified (Hindi) | [ai4bharat/sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) | CC-BY-4.0 | `verified/hin/*.parquet` (100 files, 37 GB parquet), streamed in order until the token target is met | Bulk base. Human-verified web + PDF text from AI4Bharat; the largest clean Hindi source. |
| Hindi Wikipedia | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | CC-BY-SA-3.0 / GFDL | `20231101.hi` (the only Hindi dump on the Hub; 163,093 articles), all of it | Highest-quality encyclopedic text. Upsampled 2x in train (D2). |
| Manual collection | Our own crawl from the course project (`s3/lma_minor/hindi/data/manual`, copied read-only to `raw/manual/` in the private data repo) | Crawled news articles. Copyright stays with each publisher; used only for private research training and never redistributed. The HF repo is private. | All 11 shards that pass filters | Hand-picked Hindi news sites: Dainik Jagran, Amar Ujala, Live Hindustan, Dainik Bhaskar, Prabhat Khabar, Patrika, ABP Live, BBC Hindi, … |
| FineWeb-2 (Hindi) | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ODC-BY | `data/hin_Deva/train` (22.1M docs) | Only if Sangraha runs out. Sampled in Stage 1 for comparison. **Disabled** in `configs/data.yaml` (`enabled: false`). |
| Sangraha **Synthetic** | — | — | **Excluded** | Machine-translated (D2). `data/sources.py` refuses any path containing `synthetic`. |
