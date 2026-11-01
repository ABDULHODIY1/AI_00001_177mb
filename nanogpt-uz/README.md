# nanoGPT-uz

O'zbek tili uchun noldan o'qitiladigan kichik LLM — bitta Jupyter notebookda (`nanogpt_uz.ipynb`).
Karpathy'ning nanoGPT soddaligi + Llama bloklari (RoPE, RMSNorm, SwiGLU) + DeepSeek (MLA attention,
DeepSeekMoE, aux-loss-free ekspert balansi, ixtiyoriy MTP) + Muon optimizer. JAX'da yozilgan, Google Colab
TPU v5e-1 (bepul) uchun moslangan; CPU/GPU da ham ishlaydi.

## Ishga tushirish (Colab)
1. Notebookni Colab'da oching: `File → Upload notebook`.
2. `Runtime → Change runtime type → TPU v5e-1`.
3. Sozlamalar katagida avval `PROFILE = 'smoke'` qilib **Run all** (3–5 daqiqa), keyin `PROFILE = 'real'`.
4. Google Drive ulansa, data va checkpointlar `MyDrive/nanogpt-uz` da saqlanadi. Colab uzilsa — yana **Run all**:
   trening oxirgi checkpointdan davom etadi.

Standart `real` model: 12 qatlam, d=768, MLA, 1 dense + 11 MoE qatlam (16 ekspert, top-2, 1 shared) —
~301M jami / ~94M faol parametr. Pretrain 110 daqiqa + SFT 20 daqiqa.
