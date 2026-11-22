# SpiS-GAN IAM HDF5 provenance

- Source: `DAIR-Group/SpiS-GAN` (Hugging Face model repository)
- Revision: `9a0f1b5ce46f05fb40e4c324b1cb746621ede942`
- Downloaded paths: `data/train_32.hdf5`, `data/test_32.hdf5`
- Local data directory: `data/spisgan_hdf5/` (gitignored)

Verified files:

| File | Images | Writers | SHA-256 |
|---|---:|---:|---|
| `train_32.hdf5` | 66,603 | 339 | `d31090dc037554619fdb70d53aa90597f3a69d0cb8eb1dfd833a44e99395b0d0` |
| `test_32.hdf5` | 25,766 | 161 | `5e832ec551a57a27d6824be57f1a01ce5cbf10172e4c803f8e3aa5a299b63790` |

Recreate metadata with:

```bash
python prepare_higan_hdf5.py \
  --hdf5-root data/spisgan_hdf5 \
  --dataset-format spisgan_hdf5 \
  --output data/spisgan_hdf5_metadata
```
