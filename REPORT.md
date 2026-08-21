# Báo cáo quyết định

## Giải quyết lỗi OOM ở NB3
Trong quá trình huấn luyện với Qwen3.5-9B trên BIGGPU, cấu hình mặc định đặt `max_length = 2048`.
Tuy nhiên, bước NB1 báo cáo rằng `p95 = 192` và gợi ý `max_length = 256`. Do đó, giữ `max_length = 2048` không chỉ lãng phí bộ nhớ mà còn gây ra lỗi **CUDA out of memory** (OOM) ở NB3 vì batch size và số bước padding quá lớn so với thực tế của dataset.

**Lựa chọn:** Tôi đã giảm `max_length` của cấu hình `BIGGPU` trong `src/labkit/config.py` xuống `256`. Lựa chọn này vẫn đảm bảo bao phủ >95% các chuỗi trong dataset mà không bị cắt cụt quá nhiều, đồng thời giúp giảm footprint bộ nhớ và giải quyết được lỗi OOM.
