# Neden Sadece ostrack_test.py Hata Veriyor?

Çünkü `deneme.py` ve `ostracktry.py` dosyaları **PyTorch** motorunu kullanıyor. PyTorch kendi CUDA ayarlarıyla ekran kartınızı (GPU) sorunsuz çalıştırır.

Ancak `ostrack_test.py` dosyanız **ONNX Runtime** kullanıyor. Arka planda `onnxruntime-gpu` paketi otomatik olarak **1.25.0** (en son) sürümüne güncellendi. Bu son sürüm **CUDA 12** istiyor fakat sizin sisteminizde **CUDA 11** yüklü olduğu için uyuşmazlık çıkıyor ve program çöküyor.

---

# Kesin Çözüm Adımları

Terminalde (başında `(venv)` yazarken) sırasıyla şu iki komutu çalıştırın:

**1. Hata veren ve yeni yüklenen sorunlu sürümü silin:**

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
```

**2. Sizin sisteminizle (CUDA 11) sorunsuz çalışan eski sürümü yükleyin:**

```bash
pip install onnxruntime-gpu==1.16.3
```

Bu işlemleri yaptıktan sonra `python ostrack_test.py` komutunuz eskisi gibi ekran kartı (GPU) üzerinden sorunsuz çalışacaktır!
