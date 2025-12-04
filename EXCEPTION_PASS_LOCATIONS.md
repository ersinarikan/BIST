# "except Exception: pass" Kullanımları - Detaylı Liste

Bu dosya, kod tabanında hataları sessizce yutan `except Exception: pass` kullanımlarının tam listesini içerir.

---

## 📁 scripts/show_hpo_progress.py

### 1. Satır 59-60
```python
except Exception:
    return {}
```
**Konum:** `load_state()` fonksiyonu  
**Sorun:** JSON parsing hatası sessizce yutuluyor, boş dict dönüyor  
**Etki:** Malformed JSON'lar görünmez, veri kaybı olabilir  
**Öneri:** Hata loglanmalı, en azından warning seviyesinde

---

### 2. Satır 82-83
```python
except Exception:
    pass
```
**Konum:** `get_active_hpo_processes()` - horizon parsing  
**Sorun:** Integer conversion hatası sessizce yutuluyor  
**Etki:** Yanlış horizon değerleri görmezden gelinir  
**Öneri:** Debug log eklenmeli

---

### 3. Satır 87-88
```python
except Exception:
    pass
```
**Konum:** `get_active_hpo_processes()` - trials parsing  
**Sorun:** Integer conversion hatası sessizce yutuluyor  
**Etki:** Yanlış trials değerleri görmezden gelinir  
**Öneri:** Debug log eklenmeli

---

### 4. Satır 98-99
```python
except Exception:
    pass
```
**Konum:** `get_active_hpo_processes()` - subprocess.run()  
**Sorun:** Process listeleme hatası sessizce yutuluyor  
**Etki:** Aktif HPO process'leri tespit edilemeyebilir  
**Öneri:** Error log eklenmeli

---

### 5. Satır 208-211 (2 adet)
```python
except Exception:
    pass
except Exception:
    pass
```
**Konum:** `get_trial_info_from_db()` - best_dirhit extraction (PRIORITY 1)  
**Sorun:** JSON parsing ve float conversion hataları sessizce yutuluyor  
**Etki:** Best DirHit değeri None kalabilir, hata ayıklama zorlaşır  
**Öneri:** Her seviyede debug log eklenmeli

---

### 6. Satır 234-237 (2 adet)
```python
except Exception:
    pass
except Exception:
    pass
```
**Konum:** `get_trial_info_from_db()` - best_dirhit extraction (PRIORITY 2)  
**Sorun:** JSON parsing ve float conversion hataları sessizce yutuluyor  
**Etki:** Best DirHit değeri None kalabilir  
**Öneri:** Debug log eklenmeli

---

### 7. Satır 253-256 (2 adet)
```python
except Exception:
    pass
except Exception:
    pass
```
**Konum:** `get_trial_info_from_db()` - best_dirhit extraction (PRIORITY 3)  
**Sorun:** JSON parsing ve float conversion hataları sessizce yutuluyor  
**Etki:** Best DirHit değeri None kalabilir  
**Öneri:** Debug log eklenmeli

---

### 8. Satır 271-274 (2 adet)
```python
except Exception:
    pass
except Exception:
    pass
```
**Konum:** `get_trial_info_from_db()` - best_dirhit extraction (PRIORITY 4)  
**Sorun:** JSON parsing ve float conversion hataları sessizce yutuluyor  
**Etki:** Best DirHit değeri None kalabilir  
**Öneri:** Debug log eklenmeli

---

### 9. Satır 300-303 (2 adet)
```python
except Exception:
    pass
except Exception:
    pass
```
**Konum:** `get_trial_info_from_db()` - current_dirhit extraction (PRIORITY 1)  
**Sorun:** JSON parsing ve float conversion hataları sessizce yutuluyor  
**Etki:** Current DirHit değeri None kalabilir  
**Öneri:** Debug log eklenmeli

---

### 10. Satır 326-329 (2 adet)
```python
except Exception:
    pass
except Exception:
    pass
```
**Konum:** `get_trial_info_from_db()` - current_dirhit extraction (PRIORITY 2)  
**Sorun:** JSON parsing ve float conversion hataları sessizce yutuluyor  
**Etki:** Current DirHit değeri None kalabilir  
**Öneri:** Debug log eklenmeli

---

### 11. Satır 344-347 (2 adet)
```python
except Exception:
    pass
except Exception:
    pass
```
**Konum:** `get_trial_info_from_db()` - current_dirhit extraction (PRIORITY 3)  
**Sorun:** JSON parsing ve float conversion hataları sessizce yutuluyor  
**Etki:** Current DirHit değeri None kalabilir  
**Öneri:** Debug log eklenmeli

---

### 12. Satır 362-365 (2 adet)
```python
except Exception:
    pass
except Exception:
    pass
```
**Konum:** `get_trial_info_from_db()` - current_dirhit extraction (PRIORITY 4)  
**Sorun:** JSON parsing ve float conversion hataları sessizce yutuluyor  
**Etki:** Current DirHit değeri None kalabilir  
**Öneri:** Debug log eklenmeli

---

### 13. Satır 561-562
```python
except Exception:
    continue
```
**Konum:** `get_completed_tasks()` - study file processing loop  
**Sorun:** Dosya işleme hatası sessizce yutuluyor, continue ile atlanıyor  
**Etki:** Bazı completed task'lar tespit edilemeyebilir  
**Öneri:** Error log eklenmeli, en azından warning

---

### 14. Satır 823-824
```python
except Exception:
    print(f"      📍 Güncel Trial #{current_trial} (Running - hesaplanıyor...)")
```
**Konum:** `main()` - last complete trial query  
**Sorun:** DB query hatası sessizce yutuluyor, sadece print yapılıyor  
**Etki:** Hata bilgisi kaybolur, debugging zorlaşır  
**Öneri:** Logger kullanılmalı, hata detayı loglanmalı

---

## 📁 app.py

### 15. Satır 94-96
```python
except Exception as e:
    ErrorHandler.handle(e, 'app_init_internal_token', level='debug')
    pass
```
**Konum:** `create_app()` - INTERNAL_API_TOKEN config  
**Sorun:** Hata ErrorHandler'a gönderiliyor ama sonra pass ile sessizce yutuluyor  
**Etki:** Config hatası görünmez olabilir (debug level)  
**Öneri:** En azından warning level kullanılmalı veya pass kaldırılmalı

---

### 16. Satır 107-109
```python
except Exception as e:
    ErrorHandler.handle(e, 'app_init_socketio_mq', level='debug')
    mq_url = None
```
**Konum:** `create_app()` - SOCKETIO_MESSAGE_QUEUE config  
**Sorun:** Hata debug level'da loglanıyor, production'da görünmez  
**Etki:** SocketIO message queue config hatası görünmez  
**Öneri:** Warning level kullanılmalı

---

### 17. Satır 128-129
```python
except Exception as _csrf_socketio_err:
    logger.info(f"CSRF exempt for socketio failed: {_csrf_socketio_err}")
```
**Konum:** `create_app()` - CSRF exempt  
**Not:** Bu aslında logluyor, sadece info level - sorun değil

---

### 18. Satır 136-137
```python
except Exception:
    pass
```
**Konum:** `create_app()` - template auto-reload  
**Sorun:** Template reload hatası sessizce yutuluyor  
**Etki:** Template reload çalışmayabilir, görünmez  
**Öneri:** Debug log eklenmeli

---

### 19. Satır 163-164
```python
except Exception:
    pass
```
**Konum:** `create_app()` - broadcast_log() sanitization  
**Sorun:** JSON sanitization hatası sessizce yutuluyor  
**Etki:** WebSocket emit başarısız olabilir, görünmez  
**Öneri:** Debug log eklenmeli

---

### 20. Satır 198-199
```python
except Exception:
    pass
```
**Konum:** `_start_log_tailer()` - log tailing  
**Sorun:** Log tailing hatası sessizce yutuluyor  
**Etki:** Log tailing çalışmayabilir  
**Öneri:** Error log eklenmeli

---

### 21. Satır 200-201
```python
except Exception:
    pass
```
**Konum:** `_start_log_tailer()` - _tail() function  
**Sorun:** Log tailing hatası sessizce yutuluyor  
**Etki:** Log tailing thread'i çökebilir  
**Öneri:** Error log eklenmeli

---

### 22. Satır 206-207
```python
except Exception:
    pass
```
**Konum:** `_start_log_tailer()` - thread start  
**Sorun:** Thread başlatma hatası sessizce yutuluyor  
**Etki:** Log tailing başlamayabilir  
**Öneri:** Error log eklenmeli

---

### 23. Satır 240-241
```python
except Exception as _e:
    logger.warning(f'Graceful shutdown failed: {_e}')
```
**Konum:** `_graceful_stop()` - shutdown handler  
**Not:** Bu logluyor, sorun değil

---

### 24. Satır 250-251
```python
except Exception as _e:
    logger.warning(f'Signal handler error: {_e}')
```
**Konum:** `_graceful_stop()` - signal handler  
**Not:** Bu logluyor, sorun değil

---

### 25. Satır 256-257
```python
except Exception as e:
    logger.warning(f'Signal handler setup failed: {e}')
```
**Konum:** Signal handler setup  
**Not:** Bu logluyor, sorun değil

---

### 26. Satır 296-297
```python
except Exception as e:
    logger.error(f"❌ Automation pipeline auto-start error: {e}")
```
**Konum:** `_auto_start_automation()` - delayed start  
**Not:** Bu logluyor, sorun değil

---

### 27. Satır 303-304
```python
except Exception as e:
    logger.warning(f"Auto-start automation setup failed: {e}")
```
**Konum:** `_auto_start_automation()` - setup  
**Not:** Bu logluyor, sorun değil

---

### 28. Satır 319-320
```python
except Exception:
    return False
```
**Konum:** `is_admin()` - admin check  
**Sorun:** Admin check hatası sessizce False dönüyor  
**Etki:** Admin kullanıcılar erişim alamayabilir  
**Öneri:** Error log eklenmeli

---

### 29. Satır 330-332
```python
except Exception:
    pass
```
**Konum:** `admin_required()` decorator  
**Sorun:** Admin check hatası sessizce yutuluyor  
**Etki:** Admin route'larına erişim reddedilebilir  
**Öneri:** Error log eklenmeli

---

### 30. Satır 366-367
```python
except Exception as _oauth_err:
    logger.info(f"OAuth not initialized: {_oauth_err}")
```
**Konum:** OAuth setup  
**Not:** Bu logluyor, sorun değil

---

### 31. Satır 396-397
```python
except Exception:
    pass
```
**Konum:** `internal_route()` decorator - auth check  
**Sorun:** Auth check hatası sessizce yutuluyor  
**Etki:** Internal route'lara erişim reddedilebilir, hata görünmez  
**Öneri:** Error log eklenmeli

---

### 32. Satır 403-404
```python
except Exception:
    pass
```
**Konum:** `internal_route()` - limiter.exempt()  
**Sorun:** Rate limiter exempt hatası sessizce yutuluyor  
**Etki:** Rate limiting çalışmayabilir  
**Öneri:** Debug log eklenmeli

---

### 33. Satır 407-408
```python
except Exception:
    pass
```
**Konum:** `internal_route()` - csrf.exempt()  
**Sorun:** CSRF exempt hatası sessizce yutuluyor  
**Etki:** CSRF koruması çalışmayabilir  
**Öneri:** Debug log eklenmeli

---

### 34. Satır 416-417
```python
except Exception as _cors_err:
    logger.warning(f"CORS init failed: {_cors_err}")
```
**Konum:** CORS init  
**Not:** Bu logluyor, sorun değil

---

### 35. Satır 456-457
```python
except Exception as e:
    logger.debug(f"Emit logging error: {e}")
```
**Konum:** `_logged_socketio_emit()` - emit logging  
**Not:** Bu logluyor (debug level), sorun değil

---

### 36. Satır 482-483
```python
except Exception as e:
    logger.debug(f"Status emit sanitization failed: {e}")
```
**Konum:** `handle_connect()` - status emit  
**Not:** Bu logluyor (debug level), sorun değil

---

### 37. Satır 487-488
```python
except Exception:
    pass
```
**Konum:** `handle_connect()` - fallback status emit  
**Sorun:** Fallback emit hatası sessizce yutuluyor  
**Etki:** Client'a status gönderilemeyebilir  
**Öneri:** Debug log eklenmeli

---

### 38. Satır 506-507
```python
except Exception as e:
    logger.debug(f"Room joined emit sanitization failed: {e}")
```
**Konum:** `handle_join_admin()` - room emit  
**Not:** Bu logluyor (debug level), sorun değil

---

### 39. Satır 512-513
```python
except Exception:
    pass
```
**Konum:** `handle_join_admin()` - fallback emit  
**Sorun:** Fallback emit hatası sessizce yutuluyor  
**Öneri:** Debug log eklenmeli

---

### 40. Satır 528-529
```python
except Exception as e:
    logger.debug(f"Room joined emit sanitization failed: {e}")
```
**Konum:** `handle_join_user()` - room emit  
**Not:** Bu logluyor (debug level), sorun değil

---

### 41. Satır 533-534
```python
except Exception:
    pass
```
**Konum:** `handle_join_user()` - fallback emit  
**Sorun:** Fallback emit hatası sessizce yutuluyor  
**Öneri:** Debug log eklenmeli

---

### 42. Satır 550-551
```python
except Exception as e:
    logger.debug(f"Subscription confirmed emit sanitization failed: {e}")
```
**Konum:** `handle_subscribe_stock()` - subscription emit  
**Not:** Bu logluyor (debug level), sorun değil

---

### 43. Satır 554-555
```python
except Exception:
    pass
```
**Konum:** `handle_subscribe_stock()` - fallback emit  
**Sorun:** Fallback emit hatası sessizce yutuluyor  
**Öneri:** Debug log eklenmeli

---

### 44. Satır 571-572
```python
except Exception as e:
    logger.debug(f"Subscription removed emit sanitization failed: {e}")
```
**Konum:** `handle_unsubscribe_stock()` - unsubscribe emit  
**Not:** Bu logluyor (debug level), sorun değil

---

### 45. Satır 575-577
```python
except Exception:
    pass
```
**Konum:** `handle_unsubscribe_stock()` - fallback emit  
**Sorun:** Fallback emit hatası sessizce yutuluyor  
**Öneri:** Debug log eklenmeli

---

### 46. Satır 638-643
```python
except Exception as e:
    logger.warning(f"⚠️ Atomic write failed, trying fallback: {e}")
    # Non-atomic fallback
    with open(_state_path, 'w') as wf:
        wf.write(_json.dumps(cur, ensure_ascii=False))
    logger.info(f"✅ Calibration bypass persisted (fallback) to {_state_path}")
```
**Konum:** Calibration state persistence  
**Not:** Bu logluyor ve fallback yapıyor, sorun değil

---

### 47. Satır 644-645
```python
except Exception as e:
    logger.error(f"❌ Calibration startup error: {e}")
```
**Konum:** Calibration startup  
**Not:** Bu logluyor, sorun değil

---

## 📊 ÖZET

### Toplam "except Exception: pass" Kullanımı

**scripts/show_hpo_progress.py:** 14 adet
- Satır 59-60: JSON parsing
- Satır 82-83: Horizon parsing
- Satır 87-88: Trials parsing
- Satır 98-99: Subprocess
- Satır 208-211: Best DirHit extraction (2 adet)
- Satır 234-237: Best DirHit extraction (2 adet)
- Satır 253-256: Best DirHit extraction (2 adet)
- Satır 271-274: Best DirHit extraction (2 adet)
- Satır 300-303: Current DirHit extraction (2 adet)
- Satır 326-329: Current DirHit extraction (2 adet)
- Satır 344-347: Current DirHit extraction (2 adet)
- Satır 362-365: Current DirHit extraction (2 adet)
- Satır 561-562: Study file processing
- Satır 823-824: Last complete trial query

**app.py:** 8 adet (sessizce yutan)
- Satır 94-96: Config init (ErrorHandler kullanıyor ama pass var)
- Satır 136-137: Template reload
- Satır 163-164: Broadcast log
- Satır 198-199: Log tailing
- Satır 200-201: Log tailing thread
- Satır 206-207: Thread start
- Satır 319-320: Admin check
- Satır 330-332: Admin required decorator
- Satır 396-397: Internal route auth
- Satır 403-404: Rate limiter exempt
- Satır 407-408: CSRF exempt
- Satır 487-488: Fallback emit
- Satır 512-513: Fallback emit
- Satır 533-534: Fallback emit
- Satır 554-555: Fallback emit
- Satır 575-577: Fallback emit

**Toplam:** ~22 adet sessizce yutan exception handler

---

## 🔧 ÖNERİLER

1. **Tüm `except Exception: pass` kullanımlarına en azından debug log eklenmeli**
2. **Kritik hatalar için warning/error level kullanılmalı**
3. **Spesifik exception'lar yakalanmalı (ValueError, KeyError, etc.)**
4. **Hata mesajları ve stack trace loglanmalı**

