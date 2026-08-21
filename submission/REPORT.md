# Lab 21 - Evaluation Report

**Ho ten**: Nguyen Thi Khanh Ly  **MSSV**: 2A202601403  **Ngay**: 2026-08-21
**Tier**: `BIGGPU`  **Base model**: `Qwen/Qwen3.5-9B`  **GPU thuc te**: `NVIDIA L4 22.0 GB`

Moi con so ben duoi duoc lay tu cac file trong `results/` va log chay Colab full run.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tieng Viet -> JSON triage |
| Train / val | 225 / 25, seed 42 |
| `max_length` | 192; p95 do duoc la 98 trong lan chay cuoi, suggested la 256 |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 2.0 epochs / 30 optimizer steps |

**Template co giu khoi think khong?** Co. `results/template_check.json` va log NB1 bao `reasoning preserved`, nen template khong nuot khoi `<think>`. Toi van dung `MASK_MODE=assistant-only`; doan supervised chi nam o phan assistant, khong tinh loss tren ticket nguoi dung.

Toi dat `max_length=192` thay vi 256 vi Qwen3.5-9B tren L4 bi OOM khi NB3 dung cau hinh lon hon. Gia tri 192 van nam tren p95 cua cac chuoi sau khi prompt shape moi duoc sua, va da giup training chay duoc voi `per_device_batch=1`, `grad_accum=16`, effective batch 16.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Cau tra loi nam trong loss | true |
| Cau hoi KHONG nam trong loss | true |

Doan duoc tinh loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Ket qua nay tra loi cau hoi quan trong nhat cua NB1: model khong hoc lap lai ticket hay instruction. Loss chi nam tren cau tra loi JSON cua assistant. Mode doi chung `everything` cho thay neu cau hinh sai thi ca prompt va ticket deu vao loss, supervised fraction len 100%, nen mask proof khong chi la niem tin vao flag thu vien.

---

## 3. Ba baseline (NB2 - do TRUOC khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7422 | 0.000 | 3459.8 |
| (b) base + optimized prompt | 0.820 | 0.7422 | 1.000 | 972.6 |
| (c) LoRA fine-tune | 0.990 | 0.1333 | 1.000 | 1337.2 |

**(b) co that su manh hon (a) khong?** Co. Target tang tu 0.000 len 0.820 va format tang tu 0.000 len 1.000. Toi khong sua `OPTIMIZED_PROMPT`; SHA trong `baselines_frozen.json` la `719e74d3b6232053`, trung voi ban shipped. Tu thoi diem NB2 xong, toi khong sua hai tap eval.

---

## 4. Giai phau cau hinh sai (NB4)

| Run | vi tri | r | trainable | LR | train loss (NB4) | target (NB5) | seconds | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 43,278,336 | 0.0001 | 0.2790 | 0.990 | 858.9 | 17.83 |
| `attn_only` | q,v | 311 | 43,311,104 | 0.0001 | 0.2607 | 0.995 | 733.2 | 17.84 |
| `wrong_lr` | text-linear | 16 | 43,278,336 | 0.00001 | 1.2062 | 0.000 | 877.1 | 17.83 |
| `qlora` | text-linear | 16 | 43,278,336 | 0.0001 | 0.2816 | 0.995 | 922.4 | 7.95 |

Xep hang theo target NB5: `attn_only` va `qlora` cao nhat 0.995, `correct` gan ngang 0.990, va `wrong_lr` thua hoan toan 0.000. Xep hang theo train loss lai khac: `attn_only` co loss thap nhat, `correct` va `qlora` gan nhau, `wrong_lr` cao nhat. Vi vay ket luan cua NB4 phai dua tren target, khong dua tren final_loss.

**4.1 - attn_only cung ngan sach tham so voi correct.** `attn_only` duoc match rank len r=311, trainable 43,311,104 so voi `correct` 43,278,336, sai lech rat nho. Tren target, `attn_only` dat 0.995, nho hon hoac gan ngang muc tran va nhinh hon `correct` 0.990. Thu tu nay khong chung minh rank la don bay duy nhat, vi vi tri adapter da bi thay doi nhung ngan sach tham so duoc giu cong bang. Ket qua noi voi toi rang voi bai triage ngan va co schema rat ro, q,v voi rank cao van hoc duoc hanh vi target; tuy nhien can doc cung regression/format va khong nen ket luan chi tu loss.

**4.2 - wrong_lr chi khac learning rate.** `wrong_lr` dung LR 1e-5, bang thang full fine-tune, trong khi LoRA dung 1e-4. Duong loss cua no giam cham hon rat nhieu: final_loss 1.2062 so voi 0.2790 cua correct, va target NB5 roi ve 0.000, format cung 0.000. Neu chi nhin loss ma khong biet LR, toi co the nghi model can them epoch, them rank, hoac data tot hon. Nhung doi chung nay cho thay chi mot con so LR sai da lam run gan nhu khong hoc dung hanh vi sinh JSON.

**4.3 - qlora tiet kiem VRAM nhung can do bang task.** `qlora` dung 7.95 GB peak VRAM, tiet kiem khoang 9.88 GB so voi `correct` 17.83 GB. Tren target, no dat 0.995 va format 1.000, nen trong bai ngan nay khong co gia phai tra ro rang o target. Tuy nhien ket qua nay khong du de bac bo khuyen nghi can than voi QLoRA tren Qwen3.5, vi lab chi do 50 mau target ngan va verdict chinh van that bai do regression collapse cua fine-tune. Ket luan hop ly la QLoRA tiet kiem VRAM manh, nhung can them regression va eval rong hon truoc khi chon lam cau hinh deploy.

---

## 5. Phan quyet (NB5)

**Ket qua cong hoi quy**: FAILED
`target delta = +0.170` - `regression delta = -0.609` - `valid_trace_rate = 0.0`

Fine-tune thang ro tren target: target tang tu baseline prompt tot 0.820 len 0.990, format giu 1.000. Neu chi nhin bai ticket triage, adapter co ve rat thanh cong. Nhung cong hoi quy that bai vi regression giam tu 0.7422 xuong 0.1333, tuc mat 0.609 so voi baseline (b), vuot xa tolerance 0.020. Day la dau hieu catastrophic forgetting: model hoc rat manh dang dau ra JSON cho mien CSKH, nhung kha nang tra loi cau hoi pho thong bi pha hong. Vi vay toi khong nen deploy ban fine-tune nay nhu mot assistant tong quat. Neu san pham chi dung trong backend classifier bi gioi han dau vao, ket qua target co the huu ich; nhung neu model con phai giu nang luc chung, can them 1-5% replay data pho thong hoac tach rieng model classifier. FAILED o day la ket qua co gia tri, vi no phat hien fine-tune "thang task" nhung khong an toan de dung rong.

---

## 6. Dinh tinh - bat buoc co ca ca THUA

| # | Ticket rut gon | Nhan dung | (b) prompt | (c) fine-tune | Nhan xet |
|---|---|---|---|---|---|
| 1 | Cho minh hoi, dat may xay sinh to, muon tra lai | doi_tra, product may xay sinh to | gan dung schema | score 0.75, output bi cat o sentiment | FT thua mot phan vi cau JSON/field sentiment chua tron ven trong preview |
| 2 | Alo shop, dat may xay sinh to, tra lai tien | hoan_tien hoac doi_tra tuy nhan | prompt (b) tao JSON dung format | score 0.75, urgency thap va output preview bi cat | FT thua mot phan, nham sac thai/urgency |
| 3 | Cau hoi regression pho thong | keyword answer | regression 0.7422 | regression 0.1333 | FT thua nghiem trong tren nang luc chung |
| 4 | Ticket chuot khong day, yeu cau tra lai | doi_tra, product chuot khong day | co schema day du | score 1.0 | FT thang tren target, JSON hop le |
| 5 | Op lung dien thoai, giao hang/hoi thong tin/sai mau | van_chuyen/hoi_thong_tin/san_pham_loi | co the dung prompt dai | score 1.0 | FT thang o cac ca tot nhat trong log |

Mau chung cua ca FT thua la model qua tap trung vao format JSON va mien ticket. Tren target, loi con lai thuong la field cuoi bi cat trong preview, urgency/sentiment bi lech, hoac nham intent gan nhau nhu doi_tra va hoan_tien. Tren regression, loi nang hon: model da quen nang luc tra loi pho thong, lam verdict FAILED du target rat cao.

---

## 7. Ket luan va dieu toi hoc duoc

**Ket luan.** Toi khong nen deploy ban fine-tune nay nhu mot assistant tong quat, du no thang target task rat manh. Ket qua quan trong nhat la target tang tu 0.820 cua base co prompt tot len 0.990 cua LoRA, chung minh fine-tune da dua hanh vi JSON triage vao trong adapter. Tuy nhien regression giam tu 0.7422 xuong 0.1333, nen model mat nhieu nang luc chung. Neu bai toan san pham la mot service phan loai ticket doc lap, chi nhan ticket CSKH va chi tra JSON, adapter nay co the duoc can nhac sau khi them guardrail va test them. Neu model phai tro thanh chatbot hoac xu ly cau hoi ngoai mien, toi se khong deploy. Don bay that su trong lab nay khong phai loss thap, ma la mask dung, prompt/eval dong bang, LR dung thang LoRA, va thiet ke doi chung cong bang. `wrong_lr` cho thay sai mot bac LR lam target ve 0.000. `attn_only` va `qlora` cho thay train loss khong phai thuoc do quyet dinh. Bai hoc lon nhat la fine-tune co the lam task chinh tot hon prompt engineering, nhung van that bai neu pha hong regression gate.

**Ba dieu toi hoc duoc**:
1. Mask proof phai doc bang mat. Neu supervised text co ticket nguoi dung thi loss curve dep cung vo nghia.
2. Baseline prompt tot la moc rat manh. Fine-tune chi dang gia khi thang moc nay va khong pha nang luc chung.
3. Doi chung cong bang can cung max_steps va gan cung ngan sach tham so; neu khong, ket luan ve vi tri adapter hay rank se bi lech.

**Neu co them 2 gio nua, toi se thu:** them 1-5% replay data pho thong de giam regression collapse, sau do train lai `correct` va chay NB5 de xem verdict co doi tu FAILED sang PASSED khong. Toi cung se xem raw output cua cac ca score 0.75 de tach loi do output bi cat, urgency sai, hay intent gan nhau.

---

## Phu luc - thuong da lam

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset mien rieng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quet rank co kiem soat
- [ ] B5 HuggingFace Hub
