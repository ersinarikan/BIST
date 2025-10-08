# BIST Pattern Flutter Mobil Uygulama Geliştirme Rehberi

## 📱 Kapsamlı Flutter/Dart API Entegrasyon Dokümantasyonu

Bu rehber, BIST Pattern sistemini Flutter mobil uygulamasında kullanmak için gerekli tüm API endpoint'lerini, kod örneklerini ve best practice'leri içerir.

---

## 📚 İçindekiler

1. [Gerekli Paketler](#gerekli-paketler)
2. [Proje Yapısı](#proje-yapısı)
3. [API Client Sınıfı](#api-client-sınıfı)
4. [Authentication](#authentication)
5. [Watchlist Yönetimi](#watchlist-yönetimi)
6. [Tahminler ve Analizler](#tahminler-ve-analizler)
7. [WebSocket Entegrasyonu](#websocket-entegrasyonu)
8. [State Management](#state-management)
9. [UI Bileşenleri](#ui-bileşenleri)
10. [Grafik Gösterimi](#grafik-gösterimi)
11. [Offline Destek](#offline-destek)
12. [Performance Optimizasyonu](#performance-optimizasyonu)

---

## 🔧 Gerekli Paketler

### pubspec.yaml
```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # HTTP & API
  http: ^1.1.0
  dio: ^5.3.3  # Daha gelişmiş HTTP client (önerilir)
  
  # WebSocket
  socket_io_client: ^2.0.3
  
  # State Management
  provider: ^6.1.1
  # veya
  riverpod: ^2.4.9
  # veya
  get: ^4.6.6
  
  # Local Storage
  shared_preferences: ^2.2.2
  hive: ^2.2.3  # NoSQL local database
  hive_flutter: ^1.1.0
  
  # Charts
  fl_chart: ^0.64.0
  syncfusion_flutter_charts: ^23.1.44  # Daha profesyonel (ücretli lisans gerekebilir)
  
  # UI/UX
  flutter_spinkit: ^5.2.0  # Loading animasyonları
  pull_to_refresh: ^2.0.0
  shimmer: ^3.0.0  # Skeleton loading
  cached_network_image: ^3.3.0
  
  # Utilities
  intl: ^0.18.1  # Tarih/Para formatı (Türkçe)
  timeago: ^3.6.0
  logger: ^2.0.2
  
dev_dependencies:
  flutter_test:
    sdk: flutter
  mockito: ^5.4.3  # Testing
  build_runner: ^2.4.6
```

---

## 📁 Proje Yapısı

```
lib/
├── main.dart
├── config/
│   └── api_config.dart           # API URL ve sabitler
├── models/
│   ├── stock.dart                # Hisse senedi modeli
│   ├── watchlist_item.dart       # Watchlist item modeli
│   ├── prediction.dart           # Tahmin modeli
│   ├── pattern_analysis.dart     # Analiz modeli
│   └── signal.dart               # Sinyal modeli
├── services/
│   ├── api_service.dart          # HTTP API istekleri
│   ├── websocket_service.dart    # WebSocket bağlantısı
│   ├── auth_service.dart         # Authentication
│   └── cache_service.dart        # Local cache yönetimi
├── providers/
│   ├── auth_provider.dart        # Auth state
│   ├── watchlist_provider.dart   # Watchlist state
│   └── predictions_provider.dart # Predictions state
├── screens/
│   ├── login_screen.dart
│   ├── home_screen.dart          # Ana watchlist ekranı
│   ├── stock_detail_screen.dart  # Detay ekranı
│   └── search_screen.dart        # Arama ekranı
├── widgets/
│   ├── stock_card.dart           # Hisse kartı widget
│   ├── price_chart.dart          # Fiyat grafiği
│   ├── signal_badge.dart         # Sinyal rozeti
│   └── prediction_row.dart       # Tahmin satırı
└── utils/
    ├── formatters.dart           # Para, tarih formatlama
    └── constants.dart            # Sabitler
```

---

## 🌐 API Client Sınıfı

### config/api_config.dart
```dart
class APIConfig {
  // Base URL - production'da değiştir
  static const String baseURL = 'https://your-domain.com';
  static const String apiBaseURL = '$baseURL/api';
  static const String wsURL = baseURL;
  
  // Timeout süreleri
  static const Duration connectionTimeout = Duration(seconds: 30);
  static const Duration receiveTimeout = Duration(seconds: 30);
  
  // Cache TTL (saniye)
  static const int predictionsCacheTTL = 30;
  static const int analysisCacheTTL = 60;
  static const int watchlistCacheTTL = 300;
  
  // Batch limitleri
  static const int maxBatchSymbols = 50;
  static const int maxSearchResults = 50;
}
```

### services/api_service.dart
```dart
import 'package:dio/dio.dart';
import 'package:logger/logger.dart';
import '../config/api_config.dart';
import '../models/stock.dart';
import '../models/watchlist_item.dart';
import '../models/prediction.dart';
import '../models/pattern_analysis.dart';

class APIService {
  static final APIService _instance = APIService._internal();
  factory APIService() => _instance;
  
  late Dio _dio;
  final Logger _logger = Logger();
  String? _sessionCookie;
  
  APIService._internal() {
    _dio = Dio(BaseOptions(
      baseUrl: APIConfig.apiBaseURL,
      connectTimeout: APIConfig.connectionTimeout,
      receiveTimeout: APIConfig.receiveTimeout,
      headers: {
        'Content-Type': 'application/json',
      },
      validateStatus: (status) => status! < 500,
    ));
    
    // Interceptor: Log ve hata yönetimi
    _dio.interceptors.add(InterceptorsWrapper(
      onRequest: (options, handler) {
        _logger.d('📤 ${options.method} ${options.path}');
        
        // Session cookie ekle
        if (_sessionCookie != null) {
          options.headers['Cookie'] = _sessionCookie;
        }
        
        return handler.next(options);
      },
      onResponse: (response, handler) {
        _logger.d('📥 ${response.statusCode} ${response.requestOptions.path}');
        return handler.next(response);
      },
      onError: (error, handler) {
        _logger.e('❌ API Error: ${error.message}');
        return handler.next(error);
      },
    ));
  }
  
  // Session cookie'yi kaydet
  void setSession(String cookie) {
    _sessionCookie = cookie;
  }
  
  // Session'ı temizle
  void clearSession() {
    _sessionCookie = null;
  }
  
  /// ============================================
  /// AUTHENTICATION
  /// ============================================
  
  /// Login
  Future<Map<String, dynamic>> login(String email, String password) async {
    try {
      final response = await _dio.post(
        '/login',
        data: {
          'email': email,
          'password': password,
        },
        options: Options(
          contentType: Headers.formUrlEncodedContentType,
          validateStatus: (status) => true,
        ),
      );
      
      // Session cookie'yi kaydet
      final cookies = response.headers['set-cookie'];
      if (cookies != null && cookies.isNotEmpty) {
        _sessionCookie = cookies.first;
        _logger.i('✅ Session cookie kaydedildi');
      }
      
      return {
        'success': response.statusCode == 200 || response.statusCode == 302,
        'statusCode': response.statusCode,
      };
    } catch (e) {
      _logger.e('Login error: $e');
      return {'success': false, 'error': e.toString()};
    }
  }
  
  /// Logout
  Future<void> logout() async {
    try {
      await _dio.get('/logout');
      clearSession();
    } catch (e) {
      _logger.e('Logout error: $e');
    }
  }
  
  /// ============================================
  /// WATCHLIST
  /// ============================================
  
  /// Kullanıcının watchlist'ini getir
  Future<List<WatchlistItem>> getWatchlist() async {
    try {
      final response = await _dio.get('/watchlist');
      
      if (response.statusCode == 200 && response.data['status'] == 'success') {
        final List<dynamic> items = response.data['watchlist'] ?? [];
        return items.map((json) => WatchlistItem.fromJson(json)).toList();
      }
      
      throw Exception('Watchlist yüklenemedi: ${response.data}');
    } catch (e) {
      _logger.e('Get watchlist error: $e');
      rethrow;
    }
  }
  
  /// Watchlist'e hisse ekle
  Future<WatchlistItem> addToWatchlist({
    required String symbol,
    bool alertEnabled = true,
    String? notes,
    double? alertThresholdBuy,
    double? alertThresholdSell,
  }) async {
    try {
      final response = await _dio.post(
        '/watchlist',
        data: {
          'symbol': symbol.toUpperCase(),
          'alert_enabled': alertEnabled,
          'notes': notes,
          'alert_threshold_buy': alertThresholdBuy,
          'alert_threshold_sell': alertThresholdSell,
        },
      );
      
      if (response.statusCode == 200 && response.data['status'] == 'success') {
        return WatchlistItem.fromJson(response.data['item']);
      }
      
      throw Exception('Hisse eklenemedi: ${response.data}');
    } catch (e) {
      _logger.e('Add to watchlist error: $e');
      rethrow;
    }
  }
  
  /// Watchlist'ten hisse çıkar
  Future<void> removeFromWatchlist(String symbol) async {
    try {
      final response = await _dio.delete('/watchlist/$symbol');
      
      if (response.statusCode != 200 || response.data['status'] != 'success') {
        throw Exception('Hisse çıkarılamadı: ${response.data}');
      }
    } catch (e) {
      _logger.e('Remove from watchlist error: $e');
      rethrow;
    }
  }
  
  /// ============================================
  /// PREDICTIONS (BATCH - PERFORMANSLI)
  /// ============================================
  
  /// Toplu tahmin getir (önerilen yöntem!)
  Future<Map<String, Prediction>> getBatchPredictions(List<String> symbols) async {
    try {
      if (symbols.isEmpty) return {};
      if (symbols.length > APIConfig.maxBatchSymbols) {
        throw Exception('Maksimum ${APIConfig.maxBatchSymbols} sembol gönderilebilir');
      }
      
      final response = await _dio.post(
        '/batch/predictions',
        data: {'symbols': symbols},
      );
      
      if (response.statusCode == 200 && response.data['status'] == 'success') {
        final Map<String, dynamic> results = response.data['results'] ?? {};
        final Map<String, Prediction> predictions = {};
        
        results.forEach((symbol, data) {
          if (data['status'] == 'success' || data['predictions'] != null) {
            predictions[symbol] = Prediction.fromJson(symbol, data);
          }
        });
        
        _logger.i('✅ Batch predictions: ${predictions.length} sembol');
        return predictions;
      }
      
      throw Exception('Tahminler yüklenemedi');
    } catch (e) {
      _logger.e('Batch predictions error: $e');
      rethrow;
    }
  }
  
  /// Tek sembol tahmin getir
  Future<Prediction> getUserPrediction(String symbol) async {
    try {
      final response = await _dio.get('/user/predictions/$symbol');
      
      if (response.statusCode == 200 && response.data['status'] == 'success') {
        return Prediction.fromJson(symbol, response.data);
      }
      
      throw Exception('Tahmin yüklenemedi');
    } catch (e) {
      _logger.e('Get prediction error: $e');
      rethrow;
    }
  }
  
  /// ============================================
  /// PATTERN ANALYSIS (BATCH - PERFORMANSLI)
  /// ============================================
  
  /// Toplu pattern analizi getir (önerilen yöntem!)
  Future<Map<String, PatternAnalysis>> getBatchPatternAnalysis(List<String> symbols) async {
    try {
      if (symbols.isEmpty) return {};
      if (symbols.length > APIConfig.maxBatchSymbols) {
        throw Exception('Maksimum ${APIConfig.maxBatchSymbols} sembol gönderilebilir');
      }
      
      final response = await _dio.post(
        '/batch/pattern-analysis',
        data: {'symbols': symbols},
      );
      
      if (response.statusCode == 200 && response.data['status'] == 'success') {
        final Map<String, dynamic> results = response.data['results'] ?? {};
        final Map<String, PatternAnalysis> analyses = {};
        
        results.forEach((symbol, data) {
          if (data['status'] == 'success') {
            analyses[symbol] = PatternAnalysis.fromJson(data);
          }
        });
        
        _logger.i('✅ Batch analyses: ${analyses.length} sembol');
        return analyses;
      }
      
      throw Exception('Analizler yüklenemedi');
    } catch (e) {
      _logger.e('Batch analysis error: $e');
      rethrow;
    }
  }
  
  /// Tek sembol pattern analizi getir
  Future<PatternAnalysis> getPatternAnalysis(
    String symbol, {
    bool fast = true,
  }) async {
    try {
      final response = await _dio.get(
        '/pattern-analysis/$symbol',
        queryParameters: {'fast': fast ? '1' : '0'},
      );
      
      if (response.statusCode == 200) {
        return PatternAnalysis.fromJson(response.data);
      }
      
      throw Exception('Analiz yüklenemedi');
    } catch (e) {
      _logger.e('Get pattern analysis error: $e');
      rethrow;
    }
  }
  
  /// ============================================
  /// STOCKS
  /// ============================================
  
  /// Hisse ara
  Future<List<Stock>> searchStocks(String query, {int limit = 50}) async {
    try {
      final response = await _dio.get(
        '/stocks/search',
        queryParameters: {
          'q': query,
          'limit': limit,
        },
      );
      
      if (response.statusCode == 200 && response.data['status'] == 'success') {
        final List<dynamic> stocks = response.data['stocks'] ?? [];
        return stocks.map((json) => Stock.fromJson(json)).toList();
      }
      
      return [];
    } catch (e) {
      _logger.e('Search stocks error: $e');
      return [];
    }
  }
  
  /// Hisse fiyat geçmişi getir (grafik için)
  Future<List<StockPrice>> getStockPrices(
    String symbol, {
    int days = 60,
  }) async {
    try {
      final response = await _dio.get(
        '/stock-prices/$symbol',
        queryParameters: {'days': days},
      );
      
      if (response.statusCode == 200 && response.data['status'] == 'success') {
        final List<dynamic> data = response.data['data'] ?? [];
        return data.map((json) => StockPrice.fromJson(json)).toList();
      }
      
      return [];
    } catch (e) {
      _logger.e('Get stock prices error: $e');
      return [];
    }
  }
}
```

---

## 📦 Model Sınıfları

### models/watchlist_item.dart
```dart
class WatchlistItem {
  final int id;
  final String symbol;
  final String? name;
  final String? notes;
  final bool alertEnabled;
  final double? alertThresholdBuy;
  final double? alertThresholdSell;
  final DateTime? createdAt;
  
  WatchlistItem({
    required this.id,
    required this.symbol,
    this.name,
    this.notes,
    this.alertEnabled = true,
    this.alertThresholdBuy,
    this.alertThresholdSell,
    this.createdAt,
  });
  
  factory WatchlistItem.fromJson(Map<String, dynamic> json) {
    return WatchlistItem(
      id: json['id'],
      symbol: json['symbol'],
      name: json['name'],
      notes: json['notes'],
      alertEnabled: json['alert_enabled'] ?? true,
      alertThresholdBuy: json['alert_threshold_buy']?.toDouble(),
      alertThresholdSell: json['alert_threshold_sell']?.toDouble(),
      createdAt: json['created_at'] != null 
        ? DateTime.parse(json['created_at']) 
        : null,
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'symbol': symbol,
      'name': name,
      'notes': notes,
      'alert_enabled': alertEnabled,
      'alert_threshold_buy': alertThresholdBuy,
      'alert_threshold_sell': alertThresholdSell,
      'created_at': createdAt?.toIso8601String(),
    };
  }
}
```

### models/prediction.dart
```dart
class Prediction {
  final String symbol;
  final double? currentPrice;
  final Map<String, double> predictions; // '1d': 14.03, '3d': 14.04, ...
  final Map<String, double> confidences; // '1d': 0.68, '3d': 0.67, ...
  final String? model; // 'basic' veya 'enhanced'
  final DateTime? sourceTimestamp;
  final DateTime? analysisTimestamp;
  
  Prediction({
    required this.symbol,
    this.currentPrice,
    required this.predictions,
    this.confidences = const {},
    this.model,
    this.sourceTimestamp,
    this.analysisTimestamp,
  });
  
  factory Prediction.fromJson(String symbol, Map<String, dynamic> json) {
    // Predictions map'i normalize et
    final Map<String, dynamic> rawPreds = json['predictions'] ?? {};
    final Map<String, double> preds = {};
    rawPreds.forEach((key, value) {
      if (value is num) {
        preds[key] = value.toDouble();
      }
    });
    
    // Confidences map'i normalize et
    final Map<String, dynamic> rawConfs = json['confidences'] ?? {};
    final Map<String, double> confs = {};
    rawConfs.forEach((key, value) {
      if (value is num) {
        confs[key] = value.toDouble();
      }
    });
    
    return Prediction(
      symbol: symbol,
      currentPrice: json['current_price']?.toDouble(),
      predictions: preds,
      confidences: confs,
      model: json['model'],
      sourceTimestamp: json['source_timestamp'] != null
        ? DateTime.parse(json['source_timestamp'])
        : null,
      analysisTimestamp: json['analysis_timestamp'] != null
        ? DateTime.parse(json['analysis_timestamp'])
        : null,
    );
  }
  
  // Helper: Horizon için tahmin değişim yüzdesi
  double? getChangePercent(String horizon) {
    final pred = predictions[horizon];
    if (pred == null || currentPrice == null || currentPrice == 0) {
      return null;
    }
    return ((pred - currentPrice!) / currentPrice!) * 100;
  }
  
  // Helper: Horizon için güven seviyesi
  double? getConfidence(String horizon) {
    return confidences[horizon];
  }
}
```

### models/pattern_analysis.dart
```dart
class PatternAnalysis {
  final String symbol;
  final String status;
  final DateTime timestamp;
  final double currentPrice;
  final Map<String, dynamic> indicators;
  final List<Pattern> patterns;
  final Signal overallSignal;
  final int dataPoints;
  final Map<String, MLUnified>? mlUnified;
  final bool? stale;
  final double? staleSeconds;
  
  PatternAnalysis({
    required this.symbol,
    required this.status,
    required this.timestamp,
    required this.currentPrice,
    required this.indicators,
    required this.patterns,
    required this.overallSignal,
    required this.dataPoints,
    this.mlUnified,
    this.stale,
    this.staleSeconds,
  });
  
  factory PatternAnalysis.fromJson(Map<String, dynamic> json) {
    // Patterns listesini parse et
    final List<dynamic> patternsJson = json['patterns'] ?? [];
    final List<Pattern> patterns = patternsJson
      .map((p) => Pattern.fromJson(p))
      .toList();
    
    // ML Unified parse et
    Map<String, MLUnified>? mlUnified;
    if (json['ml_unified'] != null) {
      final Map<String, dynamic> rawML = json['ml_unified'];
      mlUnified = {};
      rawML.forEach((horizon, data) {
        mlUnified![horizon] = MLUnified.fromJson(data);
      });
    }
    
    return PatternAnalysis(
      symbol: json['symbol'],
      status: json['status'] ?? 'success',
      timestamp: DateTime.parse(json['timestamp']),
      currentPrice: json['current_price']?.toDouble() ?? 0.0,
      indicators: json['indicators'] ?? {},
      patterns: patterns,
      overallSignal: Signal.fromJson(json['overall_signal'] ?? {}),
      dataPoints: json['data_points'] ?? 0,
      mlUnified: mlUnified,
      stale: json['stale'],
      staleSeconds: json['stale_seconds']?.toDouble(),
    );
  }
}

class Pattern {
  final String pattern;
  final String signal;
  final double confidence;
  final String source;
  final Map<String, int>? range;
  
  Pattern({
    required this.pattern,
    required this.signal,
    required this.confidence,
    required this.source,
    this.range,
  });
  
  factory Pattern.fromJson(Map<String, dynamic> json) {
    return Pattern(
      pattern: json['pattern'] ?? '',
      signal: json['signal'] ?? '',
      confidence: json['confidence']?.toDouble() ?? 0.0,
      source: json['source'] ?? '',
      range: json['range'] != null ? {
        'start_index': json['range']['start_index'],
        'end_index': json['range']['end_index'],
      } : null,
    );
  }
  
  // Pattern adını Türkçe'ye çevir (pattern_translations.js'den)
  String get translatedName {
    const translations = {
      'HAMMER': 'Çekiç',
      'HANGING_MAN': 'Asılan Adam',
      'DOUBLE_TOP': 'Çift Tepe',
      'DOUBLE_BOTTOM': 'Çift Dip',
      'HEAD_AND_SHOULDERS': 'Omuz Baş Omuz',
      'MARUBOZU': 'Marubozu',
      // ... diğer pattern'lar
    };
    return translations[pattern] ?? pattern.replaceAll('_', ' ');
  }
  
  // Source adını Türkçe'ye çevir
  String get sourceLabel {
    const labels = {
      'ML_PREDICTOR': 'Temel Analiz',
      'ENHANCED_ML': 'Gelişmiş Analiz',
      'VISUAL_YOLO': 'Görsel',
      'ADVANCED_TA': 'Teknik Analiz',
      'FINGPT': 'Sezgisel',
    };
    return labels[source] ?? source;
  }
}

class Signal {
  final String signal; // 'BULLISH', 'BEARISH', 'NEUTRAL'
  final double confidence;
  final int strength;
  final String reasoning;
  
  Signal({
    required this.signal,
    required this.confidence,
    required this.strength,
    required this.reasoning,
  });
  
  factory Signal.fromJson(Map<String, dynamic> json) {
    return Signal(
      signal: json['signal'] ?? 'NEUTRAL',
      confidence: json['confidence']?.toDouble() ?? 0.5,
      strength: json['strength'] ?? 50,
      reasoning: json['reasoning'] ?? '',
    );
  }
  
  // Sinyal etiketini Türkçe'ye çevir
  String get label {
    if (signal == 'BULLISH') {
      if (confidence >= 0.85) return 'Yüksek Alım Sinyali';
      if (confidence >= 0.70) return 'Alım Sinyali';
      if (confidence >= 0.55) return 'Zayıf Alım';
      return 'Bekleme';
    } else if (signal == 'BEARISH') {
      if (confidence >= 0.85) return 'Yüksek Satış Sinyali';
      if (confidence >= 0.70) return 'Satış Sinyali';
      if (confidence >= 0.55) return 'Zayıf Satış';
      return 'Bekleme';
    }
    return 'Bekleme';
  }
  
  // Sinyal rengi
  Color get color {
    if (signal == 'BULLISH' && confidence >= 0.55) {
      return Colors.green;
    } else if (signal == 'BEARISH' && confidence >= 0.55) {
      return Colors.red;
    }
    return Colors.grey;
  }
}

class MLUnified {
  final MLModel? basic;
  final MLModel? enhanced;
  final String? best;
  
  MLUnified({
    this.basic,
    this.enhanced,
    this.best,
  });
  
  factory MLUnified.fromJson(Map<String, dynamic> json) {
    return MLUnified(
      basic: json['basic'] != null ? MLModel.fromJson(json['basic']) : null,
      enhanced: json['enhanced'] != null ? MLModel.fromJson(json['enhanced']) : null,
      best: json['best'],
    );
  }
  
  // En iyi modeli döndür
  MLModel? get bestModel {
    if (best == 'enhanced') return enhanced;
    if (best == 'basic') return basic;
    return enhanced ?? basic;
  }
}

class MLModel {
  final double price;
  final double? confidence;
  final double? deltaPct;
  final double? reliability;
  final Evidence? evidence;
  
  MLModel({
    required this.price,
    this.confidence,
    this.deltaPct,
    this.reliability,
    this.evidence,
  });
  
  factory MLModel.fromJson(Map<String, dynamic> json) {
    return MLModel(
      price: json['price']?.toDouble() ?? 0.0,
      confidence: json['confidence']?.toDouble(),
      deltaPct: json['delta_pct']?.toDouble(),
      reliability: json['reliability']?.toDouble(),
      evidence: json['evidence'] != null 
        ? Evidence.fromJson(json['evidence']) 
        : null,
    );
  }
}

class Evidence {
  final double? patternScore;
  final double? sentimentScore;
  final double? contribConf;
  final double? wPat;
  final double? wSent;
  final double? boosterProb;
  final double? contribBooster;
  final double? contribDelta;
  final String? source;
  
  Evidence({
    this.patternScore,
    this.sentimentScore,
    this.contribConf,
    this.wPat,
    this.wSent,
    this.boosterProb,
    this.contribBooster,
    this.contribDelta,
    this.source,
  });
  
  factory Evidence.fromJson(Map<String, dynamic> json) {
    return Evidence(
      patternScore: json['pattern_score']?.toDouble(),
      sentimentScore: json['sentiment_score']?.toDouble(),
      contribConf: json['contrib_conf']?.toDouble(),
      wPat: json['w_pat']?.toDouble(),
      wSent: json['w_sent']?.toDouble(),
      boosterProb: json['booster_prob']?.toDouble(),
      contribBooster: json['contrib_booster']?.toDouble(),
      contribDelta: json['contrib_delta']?.toDouble(),
      source: json['source'],
    );
  }
  
  // Evidence özet metnini oluştur
  String getSummary() {
    final parts = <String>[];
    
    if (patternScore != null) {
      final sign = patternScore! >= 0 ? '+' : '';
      parts.add('Pat $sign${patternScore!.toStringAsFixed(2)}');
    }
    
    if (sentimentScore != null) {
      final sign = sentimentScore! >= 0 ? '+' : '';
      parts.add('Sent $sign${sentimentScore!.toStringAsFixed(2)}');
    }
    
    if (contribConf != null && contribConf!.abs() > 0.001) {
      final sign = contribConf! >= 0 ? '+' : '';
      parts.add('Δgüv $sign${(contribConf! * 100).toStringAsFixed(0)}');
    }
    
    if (wPat != null && wSent != null) {
      parts.add('w_pat=${wPat!.toStringAsFixed(2)}, w_sent=${wSent!.toStringAsFixed(2)}');
    }
    
    return parts.join(' | ');
  }
}
```

### models/stock.dart
```dart
class Stock {
  final int id;
  final String symbol;
  final String name;
  final String sector;
  final double? price;
  final DateTime? lastUpdate;
  
  Stock({
    required this.id,
    required this.symbol,
    required this.name,
    required this.sector,
    this.price,
    this.lastUpdate,
  });
  
  factory Stock.fromJson(Map<String, dynamic> json) {
    return Stock(
      id: json['id'],
      symbol: json['symbol'],
      name: json['name'] ?? json['symbol'],
      sector: json['sector'] ?? 'Bilinmiyor',
      price: json['price']?.toDouble(),
      lastUpdate: json['last_update'] != null
        ? DateTime.parse(json['last_update'])
        : null,
    );
  }
}

class StockPrice {
  final DateTime date;
  final double open;
  final double high;
  final double low;
  final double close;
  final int volume;
  
  StockPrice({
    required this.date,
    required this.open,
    required this.high,
    required this.low,
    required this.close,
    required this.volume,
  });
  
  factory StockPrice.fromJson(Map<String, dynamic> json) {
    return StockPrice(
      date: DateTime.parse(json['date']),
      open: json['open']?.toDouble() ?? 0.0,
      high: json['high']?.toDouble() ?? 0.0,
      low: json['low']?.toDouble() ?? 0.0,
      close: json['close']?.toDouble() ?? 0.0,
      volume: json['volume'] ?? 0,
    );
  }
}
```

---

## 🔌 WebSocket Service

### services/websocket_service.dart
```dart
import 'package:socket_io_client/socket_io_client.dart' as IO;
import 'package:logger/logger.dart';
import 'dart:async';

class WebSocketService {
  static final WebSocketService _instance = WebSocketService._internal();
  factory WebSocketService() => _instance;
  
  IO.Socket? _socket;
  final Logger _logger = Logger();
  bool _isConnected = false;
  
  // Event stream controllers
  final _patternAnalysisController = StreamController<Map<String, dynamic>>.broadcast();
  final _liveSignalController = StreamController<Map<String, dynamic>>.broadcast();
  final _connectionController = StreamController<bool>.broadcast();
  
  // Getters for streams
  Stream<Map<String, dynamic>> get patternAnalysisStream => _patternAnalysisController.stream;
  Stream<Map<String, dynamic>> get liveSignalStream => _liveSignalController.stream;
  Stream<bool> get connectionStream => _connectionController.stream;
  
  WebSocketService._internal();
  
  /// WebSocket bağlantısını başlat
  void connect(String baseURL, int userId) {
    if (_socket != null && _socket!.connected) {
      _logger.w('WebSocket zaten bağlı');
      return;
    }
    
    _logger.i('🔌 WebSocket bağlanıyor: $baseURL');
    
    _socket = IO.io(baseURL, <String, dynamic>{
      'path': '/socket.io',
      'transports': ['websocket', 'polling'],
      'autoConnect': true,
      'reconnection': true,
      'reconnectionDelay': 1500,
      'reconnectionAttempts': 8,
      'timeout': 20000,
    });
    
    // Event listeners
    _socket!.on('connect', (_) {
      _isConnected = true;
      _logger.i('✅ WebSocket bağlandı: ${_socket!.id}');
      _connectionController.add(true);
      
      // Kullanıcı odasına katıl
      _socket!.emit('join_user', {'user_id': userId});
    });
    
    _socket!.on('disconnect', (_) {
      _isConnected = false;
      _logger.w('❌ WebSocket bağlantısı kesildi');
      _connectionController.add(false);
    });
    
    _socket!.on('connect_error', (error) {
      _logger.e('❌ WebSocket bağlantı hatası: $error');
      _connectionController.add(false);
    });
    
    _socket!.on('room_joined', (data) {
      _logger.i('👤 Odaya katıldı: ${data['room']}');
    });
    
    _socket!.on('pattern_analysis', (data) {
      _logger.d('📊 Pattern analizi güncellendi: ${data['symbol']}');
      _patternAnalysisController.add(data);
    });
    
    _socket!.on('user_signal', (data) {
      _logger.i('🔔 Canlı sinyal: ${data['signal']?['symbol']}');
      _liveSignalController.add(data);
    });
    
    _socket!.on('subscription_confirmed', (data) {
      _logger.d('✅ Subscribe edildi: ${data['symbol']}');
    });
    
    _socket!.on('error', (data) {
      _logger.e('❌ WebSocket hatası: ${data['message']}');
    });
  }
  
  /// Hisseye subscribe ol
  void subscribeToStock(String symbol) {
    if (!_isConnected || _socket == null) {
      _logger.w('WebSocket bağlı değil, subscribe edilemiyor: $symbol');
      return;
    }
    _socket!.emit('subscribe_stock', {'symbol': symbol.toUpperCase()});
  }
  
  /// Hisseden unsubscribe ol
  void unsubscribeFromStock(String symbol) {
    if (!_isConnected || _socket == null) return;
    _socket!.emit('unsubscribe_stock', {'symbol': symbol.toUpperCase()});
  }
  
  /// Çoklu hisseye subscribe ol
  void subscribeToMultiple(List<String> symbols) {
    symbols.forEach((symbol) => subscribeToStock(symbol));
  }
  
  /// Bağlantıyı kes
  void disconnect() {
    _socket?.disconnect();
    _socket?.dispose();
    _socket = null;
    _isConnected = false;
  }
  
  /// Bağlantı durumu
  bool get isConnected => _isConnected;
  
  /// Temizlik (dispose)
  void dispose() {
    disconnect();
    _patternAnalysisController.close();
    _liveSignalController.close();
    _connectionController.close();
  }
}
```

---

## 🎨 UI Widgets

### widgets/stock_card.dart
```dart
import 'package:flutter/material.dart';
import '../models/watchlist_item.dart';
import '../models/prediction.dart';
import '../models/pattern_analysis.dart';
import '../utils/formatters.dart';

class StockCard extends StatelessWidget {
  final WatchlistItem watchlistItem;
  final Prediction? prediction;
  final PatternAnalysis? analysis;
  final VoidCallback onTap;
  final VoidCallback onRemove;
  
  const StockCard({
    Key? key,
    required this.watchlistItem,
    this.prediction,
    this.analysis,
    required this.onTap,
    required this.onRemove,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Colors.blue, width: 2),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Başlık satırı
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        watchlistItem.symbol,
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      if (watchlistItem.name != null)
                        Text(
                          watchlistItem.name!,
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[600],
                          ),
                        ),
                    ],
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      // Fiyat
                      Text(
                        Formatters.currency(prediction?.currentPrice),
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      // Sinyal
                      if (analysis?.overallSignal != null)
                        _buildSignalChip(analysis!.overallSignal),
                    ],
                  ),
                ],
              ),
              
              SizedBox(height: 12),
              
              // Tahminler satırı (1G/3G/7G/14G/30G)
              if (prediction != null)
                _buildPredictionsRow(prediction!),
              
              SizedBox(height: 8),
              
              // Rozetler (pattern badges)
              if (analysis != null)
                _buildPatternBadges(analysis!),
              
              SizedBox(height: 8),
              
              // Aksiyonlar
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  IconButton(
                    icon: Icon(Icons.delete_outline, color: Colors.red),
                    onPressed: onRemove,
                    tooltip: 'Takipten Çıkar',
                  ),
                  IconButton(
                    icon: Icon(Icons.list_alt, color: Colors.blue),
                    onPressed: onTap,
                    tooltip: 'Detay',
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildSignalChip(Signal signal) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: signal.color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: signal.color, width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            signal.signal == 'BULLISH' ? Icons.arrow_upward :
            signal.signal == 'BEARISH' ? Icons.arrow_downward :
            Icons.remove,
            size: 14,
            color: signal.color,
          ),
          SizedBox(width: 4),
          Text(
            '${signal.label} (%${(signal.confidence * 100).toInt()})',
            style: TextStyle(
              fontSize: 12,
              color: signal.color,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildPredictionsRow(Prediction pred) {
    final horizons = ['1d', '3d', '7d', '14d', '30d'];
    final selectedHorizon = '7d'; // Varsayılan
    
    return Wrap(
      spacing: 8,
      runSpacing: 4,
      children: [
        ...horizons.map((h) {
          final price = pred.predictions[h];
          return Text(
            '${h.toUpperCase().replaceAll('D', 'G')}: ${Formatters.currency(price)}',
            style: TextStyle(fontSize: 11),
          );
        }).toList(),
        // Seçili horizon için değişim yüzdesi
        () {
          final changePct = pred.getChangePercent(selectedHorizon);
          if (changePct == null) return SizedBox.shrink();
          
          return Container(
            padding: EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: changePct >= 0 ? Colors.green.shade50 : Colors.red.shade50,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              'Seçili ${selectedHorizon.toUpperCase()}: ${Formatters.percentage(changePct)}',
              style: TextStyle(
                fontSize: 11,
                color: changePct >= 0 ? Colors.green : Colors.red,
                fontWeight: FontWeight.bold,
              ),
            ),
          );
        }(),
      ],
    );
  }
  
  Widget _buildPatternBadges(PatternAnalysis analysis) {
    // ML unified'dan rozet oluştur
    final badges = <Widget>[];
    final selectedHorizon = '7d';
    
    if (analysis.mlUnified != null && analysis.mlUnified!.containsKey(selectedHorizon)) {
      final mlData = analysis.mlUnified![selectedHorizon]!;
      final best = mlData.best;
      
      if (mlData.enhanced != null) {
        badges.add(_buildBadge(
          'Gelişmiş $selectedHorizon',
          Colors.orange,
          isBold: best == 'enhanced',
        ));
      }
      
      if (mlData.basic != null) {
        badges.add(_buildBadge(
          'Temel $selectedHorizon',
          Colors.blue,
          isBold: best == 'basic',
        ));
      }
    }
    
    // Teknik/Görsel pattern rozetleri ekle
    final techPatterns = analysis.patterns.where((p) => 
      !['ML_PREDICTOR', 'ENHANCED_ML'].contains(p.source)
    ).take(4).toList();
    
    techPatterns.forEach((pattern) {
      final color = _getPatternColor(pattern.source);
      badges.add(_buildBadge(
        pattern.translatedName,
        color,
      ));
    });
    
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      children: badges.take(6).toList(),
    );
  }
  
  Widget _buildBadge(String label, Color color, {bool isBold = false}) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: color,
          width: isBold ? 2 : 1,
        ),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 10,
          color: color.shade800,
          fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
        ),
      ),
    );
  }
  
  Color _getPatternColor(String source) {
    switch (source.toUpperCase()) {
      case 'VISUAL_YOLO':
        return Colors.purple;
      case 'ADVANCED_TA':
        return Colors.red;
      case 'FINGPT':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }
}
```

---

## 📊 Ana Ekran (Home Screen)

### screens/home_screen.dart
```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/watchlist_provider.dart';
import '../widgets/stock_card.dart';
import 'stock_detail_screen.dart';
import 'search_screen.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isLoading = false;
  
  @override
  void initState() {
    super.initState();
    _loadData();
  }
  
  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    
    try {
      final provider = context.read<WatchlistProvider>();
      await provider.loadWatchlist();
      await provider.loadBatchData();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Veri yüklenemedi: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Icon(Icons.trending_up),
            SizedBox(width: 8),
            Text('BIST AI Hisse Takip'),
          ],
        ),
        actions: [
          // WebSocket durum göstergesi
          Consumer<WatchlistProvider>(
            builder: (context, provider, child) {
              return Icon(
                Icons.wifi,
                color: provider.isWebSocketConnected ? Colors.green : Colors.red,
              );
            },
          ),
          SizedBox(width: 16),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadData,
        child: _isLoading
          ? Center(child: CircularProgressIndicator())
          : Consumer<WatchlistProvider>(
              builder: (context, provider, child) {
                if (provider.watchlist.isEmpty) {
                  return _buildEmptyState();
                }
                
                return ListView.builder(
                  itemCount: provider.watchlist.length,
                  itemBuilder: (context, index) {
                    final item = provider.watchlist[index];
                    final prediction = provider.predictions[item.symbol];
                    final analysis = provider.analyses[item.symbol];
                    
                    return StockCard(
                      watchlistItem: item,
                      prediction: prediction,
                      analysis: analysis,
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => StockDetailScreen(
                              symbol: item.symbol,
                            ),
                          ),
                        );
                      },
                      onRemove: () => _confirmRemove(item.symbol),
                    );
                  },
                );
              },
            ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          final result = await Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => SearchScreen()),
          );
          
          if (result != null) {
            await _loadData();
          }
        },
        child: Icon(Icons.add),
        tooltip: 'Hisse Ekle',
      ),
    );
  }
  
  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.bar_chart, size: 80, color: Colors.grey),
          SizedBox(height: 16),
          Text(
            'Henüz takip edilen hisse yok',
            style: TextStyle(fontSize: 18, color: Colors.grey),
          ),
          SizedBox(height: 8),
          Text(
            'Hisse eklemek için + butonuna dokunun',
            style: TextStyle(fontSize: 14, color: Colors.grey[600]),
          ),
        ],
      ),
    );
  }
  
  Future<void> _confirmRemove(String symbol) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Emin misiniz?'),
        content: Text('$symbol takipten çıkarılsın mı?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: Text('İptal'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text('Çıkar', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
    
    if (confirm == true) {
      try {
        await context.read<WatchlistProvider>().removeFromWatchlist(symbol);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('$symbol takipten çıkarıldı')),
        );
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Hata: $e')),
        );
      }
    }
  }
}
```

---

## 🔐 State Management (Provider)

### providers/watchlist_provider.dart
```dart
import 'package:flutter/foundation.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';
import '../models/watchlist_item.dart';
import '../models/prediction.dart';
import '../models/pattern_analysis.dart';

class WatchlistProvider with ChangeNotifier {
  final APIService _api = APIService();
  final WebSocketService _ws = WebSocketService();
  
  List<WatchlistItem> _watchlist = [];
  Map<String, Prediction> _predictions = {};
  Map<String, PatternAnalysis> _analyses = {};
  bool _isLoading = false;
  bool _wsConnected = false;
  
  // Getters
  List<WatchlistItem> get watchlist => _watchlist;
  Map<String, Prediction> get predictions => _predictions;
  Map<String, PatternAnalysis> get analyses => _analyses;
  bool get isLoading => _isLoading;
  bool get isWebSocketConnected => _wsConnected;
  
  WatchlistProvider() {
    _initializeWebSocket();
  }
  
  /// WebSocket'i başlat
  void _initializeWebSocket() {
    // Connection durumunu dinle
    _ws.connectionStream.listen((connected) {
      _wsConnected = connected;
      notifyListeners();
      
      if (connected) {
        // Bağlantı kurulunca tüm watchlist'e subscribe ol
        _ws.subscribeToMultiple(_watchlist.map((w) => w.symbol).toList());
      }
    });
    
    // Pattern analizi güncellemelerini dinle
    _ws.patternAnalysisStream.listen((data) {
      final symbol = data['symbol'] as String?;
      if (symbol != null && _watchlist.any((w) => w.symbol == symbol)) {
        // Analizi güncelle
        final analysis = PatternAnalysis.fromJson(data['data']);
        _analyses[symbol] = analysis;
        notifyListeners();
      }
    });
    
    // Canlı sinyalleri dinle
    _ws.liveSignalStream.listen((data) {
      // Bildirim göster
      _showLiveSignalNotification(data);
    });
  }
  
  /// Watchlist'i yükle
  Future<void> loadWatchlist() async {
    try {
      _isLoading = true;
      notifyListeners();
      
      _watchlist = await _api.getWatchlist();
      
      _isLoading = false;
      notifyListeners();
    } catch (e) {
      _isLoading = false;
      notifyListeners();
      rethrow;
    }
  }
  
  /// Batch data yükle (predictions + analyses)
  Future<void> loadBatchData() async {
    if (_watchlist.isEmpty) return;
    
    try {
      final symbols = _watchlist.map((w) => w.symbol).toList();
      
      // Paralel olarak hem tahminleri hem analizleri çek
      final results = await Future.wait([
        _api.getBatchPredictions(symbols),
        _api.getBatchPatternAnalysis(symbols),
      ]);
      
      _predictions = results[0] as Map<String, Prediction>;
      _analyses = results[1] as Map<String, PatternAnalysis>;
      
      notifyListeners();
    } catch (e) {
      debugPrint('Batch data load error: $e');
      rethrow;
    }
  }
  
  /// Hisse ekle
  Future<void> addToWatchlist(String symbol) async {
    try {
      final item = await _api.addToWatchlist(symbol: symbol);
      _watchlist.add(item);
      
      // WebSocket'e subscribe ol
      _ws.subscribeToStock(symbol);
      
      // Yeni hisse için data yükle
      await loadBatchData();
      
      notifyListeners();
    } catch (e) {
      debugPrint('Add to watchlist error: $e');
      rethrow;
    }
  }
  
  /// Hisse çıkar
  Future<void> removeFromWatchlist(String symbol) async {
    try {
      await _api.removeFromWatchlist(symbol);
      _watchlist.removeWhere((w) => w.symbol == symbol);
      _predictions.remove(symbol);
      _analyses.remove(symbol);
      
      // WebSocket'ten unsubscribe ol
      _ws.unsubscribeFromStock(symbol);
      
      notifyListeners();
    } catch (e) {
      debugPrint('Remove from watchlist error: $e');
      rethrow;
    }
  }
  
  /// WebSocket bağlantısını başlat
  void connectWebSocket(String baseURL, int userId) {
    _ws.connect(baseURL, userId);
  }
  
  /// Temizlik
  @override
  void dispose() {
    _ws.dispose();
    super.dispose();
  }
  
  void _showLiveSignalNotification(Map<String, dynamic> data) {
    // Bu fonksiyon notification service ile entegre edilebilir
    debugPrint('🔔 Canlı sinyal: ${data['signal']?['symbol']}');
  }
}
```

---

## 🎨 Detay Ekranı

### screens/stock_detail_screen.dart
```dart
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../services/api_service.dart';
import '../models/pattern_analysis.dart';
import '../models/stock.dart';
import '../utils/formatters.dart';

class StockDetailScreen extends StatefulWidget {
  final String symbol;
  
  const StockDetailScreen({Key? key, required this.symbol}) : super(key: key);
  
  @override
  _StockDetailScreenState createState() => _StockDetailScreenState();
}

class _StockDetailScreenState extends State<StockDetailScreen> {
  final APIService _api = APIService();
  
  PatternAnalysis? _analysis;
  List<StockPrice>? _priceHistory;
  bool _isLoading = true;
  
  @override
  void initState() {
    super.initState();
    _loadData();
  }
  
  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    
    try {
      // Paralel olarak hem analizi hem fiyat geçmişini yükle
      final results = await Future.wait([
        _api.getPatternAnalysis(widget.symbol, fast: true),
        _api.getStockPrices(widget.symbol, days: 60),
      ]);
      
      setState(() {
        _analysis = results[0] as PatternAnalysis;
        _priceHistory = results[1] as List<StockPrice>;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Veri yüklenemedi: $e')),
      );
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('${widget.symbol} Detay'),
      ),
      body: _isLoading
        ? Center(child: CircularProgressIndicator())
        : SingleChildScrollView(
            padding: EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Fiyat ve genel sinyal
                _buildPriceHeader(),
                
                SizedBox(height: 16),
                
                // Grafik
                _buildChart(),
                
                SizedBox(height: 16),
                
                // Formasyonlar
                _buildPatternsSection(),
                
                SizedBox(height: 16),
                
                // ML Özet (Birleşik)
                _buildMLSummarySection(),
                
                SizedBox(height: 16),
                
                // Teknik göstergeler
                _buildIndicatorsSection(),
              ],
            ),
          ),
    );
  }
  
  Widget _buildPriceHeader() {
    if (_analysis == null) return SizedBox.shrink();
    
    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  widget.symbol,
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                ),
                Text(
                  Formatters.currency(_analysis!.currentPrice),
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            SizedBox(height: 12),
            _buildSignalCard(_analysis!.overallSignal),
          ],
        ),
      ),
    );
  }
  
  Widget _buildSignalCard(Signal signal) {
    return Container(
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: signal.color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: signal.color, width: 2),
      ),
      child: Row(
        children: [
          Icon(
            signal.signal == 'BULLISH' ? Icons.trending_up :
            signal.signal == 'BEARISH' ? Icons.trending_down :
            Icons.trending_flat,
            size: 32,
            color: signal.color,
          ),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  signal.label,
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: signal.color,
                  ),
                ),
                Text(
                  'Güven: %${(signal.confidence * 100).toInt()}',
                  style: TextStyle(fontSize: 14, color: Colors.grey[700]),
                ),
                Text(
                  signal.reasoning,
                  style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildChart() {
    if (_priceHistory == null || _priceHistory!.isEmpty) {
      return Card(
        child: Container(
          height: 250,
          child: Center(child: Text('Grafik verisi yok')),
        ),
      );
    }
    
    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Fiyat Grafiği (60 Gün)',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            Container(
              height: 250,
              child: LineChart(
                LineChartData(
                  gridData: FlGridData(show: true, drawVerticalLine: false),
                  titlesData: FlTitlesData(
                    leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    rightTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 50,
                        getTitlesWidget: (value, meta) {
                          return Text(
                            Formatters.currency(value),
                            style: TextStyle(fontSize: 10),
                          );
                        },
                      ),
                    ),
                    topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  ),
                  borderData: FlBorderData(show: false),
                  lineBarsData: [
                    // Ana fiyat çizgisi
                    LineChartBarData(
                      spots: _priceHistory!.asMap().entries.map((entry) {
                        return FlSpot(entry.key.toDouble(), entry.value.close);
                      }).toList(),
                      isCurved: true,
                      color: Colors.blue,
                      barWidth: 2,
                      dotData: FlDotData(show: false),
                      belowBarData: BarAreaData(
                        show: true,
                        color: Colors.blue.withOpacity(0.1),
                      ),
                    ),
                    // TODO: Pattern overlay'ler (kırmızı vurgular)
                  ],
                ),
              ),
            ),
            SizedBox(height: 8),
            _buildChartStats(),
          ],
        ),
      ),
    );
  }
  
  Widget _buildChartStats() {
    if (_priceHistory == null || _priceHistory!.isEmpty) return SizedBox.shrink();
    
    final prices = _priceHistory!.map((p) => p.close).toList();
    final min = prices.reduce((a, b) => a < b ? a : b);
    final max = prices.reduce((a, b) => a > b ? a : b);
    
    return Text(
      'Bar: ${prices.length} • En düşük: ${Formatters.currency(min)} • En yüksek: ${Formatters.currency(max)}',
      style: TextStyle(fontSize: 12, color: Colors.grey[600]),
    );
  }
  
  Widget _buildPatternsSection() {
    if (_analysis == null || _analysis!.patterns.isEmpty) {
      return SizedBox.shrink();
    }
    
    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Tespit Edilen Formasyonlar',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 12),
            ..._analysis!.patterns.take(10).map((pattern) {
              return Padding(
                padding: EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    Container(
                      padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: _getPatternColor(pattern.source).withOpacity(0.2),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        pattern.sourceLabel,
                        style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold),
                      ),
                    ),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        pattern.translatedName,
                        style: TextStyle(fontSize: 14),
                      ),
                    ),
                    Text(
                      '%${(pattern.confidence * 100).toInt()}',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }
  
  Widget _buildMLSummarySection() {
    if (_analysis == null || _analysis!.mlUnified == null) {
      return SizedBox.shrink();
    }
    
    final mlUnified = _analysis!.mlUnified!;
    final horizons = ['1d', '3d', '7d', '14d', '30d'];
    
    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'ML Özet (Birleşik)',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 12),
            ...horizons.map((h) {
              if (!mlUnified.containsKey(h)) {
                return Padding(
                  padding: EdgeInsets.only(bottom: 12),
                  child: Text('${h.toUpperCase()}: -'),
                );
              }
              
              final unified = mlUnified[h]!;
              return _buildMLHorizonDetail(h, unified);
            }).toList(),
          ],
        ),
      ),
    );
  }
  
  Widget _buildMLHorizonDetail(String horizon, MLUnified unified) {
    return Padding(
      padding: EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            horizon.toUpperCase(),
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 4),
          
          // Basic model
          if (unified.basic != null)
            _buildModelDetail('Temel', unified.basic!, Colors.blue),
          
          SizedBox(height: 4),
          
          // Enhanced model
          if (unified.enhanced != null)
            _buildModelDetail('Gelişmiş', unified.enhanced!, Colors.orange),
          
          SizedBox(height: 4),
          
          // En iyi model rozeti
          if (unified.best != null)
            Container(
              padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                'En iyi: ${unified.best == 'enhanced' ? 'Gelişmiş' : 'Temel'}',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold),
              ),
            ),
        ],
      ),
    );
  }
  
  Widget _buildModelDetail(String label, MLModel model, Color color) {
    final changePct = model.deltaPct != null ? model.deltaPct! * 100 : null;
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              '$label: ',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
            ),
            Text(
              Formatters.currency(model.price),
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
            ),
            SizedBox(width: 8),
            if (changePct != null)
              Text(
                Formatters.percentage(changePct),
                style: TextStyle(
                  fontSize: 13,
                  color: changePct >= 0 ? Colors.green : Colors.red,
                ),
              ),
            if (model.confidence != null)
              Text(
                ' • Güven %${(model.confidence! * 100).toInt()}',
                style: TextStyle(fontSize: 12, color: Colors.grey[600]),
              ),
          ],
        ),
        
        // Evidence detayları
        if (model.evidence != null && model.evidence!.getSummary().isNotEmpty)
          Padding(
            padding: EdgeInsets.only(left: 16, top: 4),
            child: Text(
              '→ Kanıt: ${model.evidence!.getSummary()}',
              style: TextStyle(fontSize: 11, color: Colors.grey[600]),
            ),
          ),
      ],
    );
  }
  
  Widget _buildIndicatorsSection() {
    if (_analysis == null) return SizedBox.shrink();
    
    final indicators = _analysis!.indicators;
    
    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Teknik Göstergeler',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 12),
            _buildIndicatorRow('RSI', indicators['rsi']),
            _buildIndicatorRow('MACD', indicators['macd']),
            _buildIndicatorRow('SMA 20', indicators['sma_20']),
            _buildIndicatorRow('SMA 50', indicators['sma_50']),
            _buildIndicatorRow('Bollinger Üst', indicators['bb_upper']),
            _buildIndicatorRow('Bollinger Alt', indicators['bb_lower']),
          ],
        ),
      ),
    );
  }
  
  Widget _buildIndicatorRow(String label, dynamic value) {
    return Padding(
      padding: EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(fontSize: 14)),
          Text(
            value != null ? value.toStringAsFixed(2) : '-',
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
  
  Color _getPatternColor(String source) {
    switch (source.toUpperCase()) {
      case 'VISUAL_YOLO': return Colors.purple;
      case 'ADVANCED_TA': return Colors.red;
      case 'FINGPT': return Colors.green;
      default: return Colors.grey;
    }
  }
}
```

---

## 🛠️ Utility Fonksiyonlar

### utils/formatters.dart
```dart
import 'package:intl/intl.dart';

class Formatters {
  // Türk Lirası formatı
  static final _currencyFormat = NumberFormat.currency(
    locale: 'tr_TR',
    symbol: '₺',
    decimalDigits: 2,
  );
  
  // Yüzde formatı
  static final _percentFormat = NumberFormat.percentPattern('tr_TR');
  
  /// Para formatı
  static String currency(double? value) {
    if (value == null) return '-';
    return _currencyFormat.format(value);
  }
  
  /// Yüzde formatı
  static String percentage(double? value) {
    if (value == null) return '-';
    final sign = value >= 0 ? '+' : '';
    return '$sign${value.toStringAsFixed(1)}%';
  }
  
  /// Tarih formatı
  static String date(DateTime? date) {
    if (date == null) return '-';
    return DateFormat('dd.MM.yyyy', 'tr_TR').format(date);
  }
  
  /// Tarih ve saat formatı
  static String dateTime(DateTime? dateTime) {
    if (dateTime == null) return '-';
    return DateFormat('dd.MM.yyyy HH:mm', 'tr_TR').format(dateTime);
  }
  
  /// Zaman farkı (timeago)
  static String timeAgo(DateTime? dateTime) {
    if (dateTime == null) return '-';
    
    final difference = DateTime.now().difference(dateTime);
    
    if (difference.inSeconds < 60) {
      return '${difference.inSeconds} saniye önce';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes} dakika önce';
    } else if (difference.inHours < 24) {
      return '${difference.inHours} saat önce';
    } else {
      return '${difference.inDays} gün önce';
    }
  }
  
  /// Hacim formatı (binlik ayraçlı)
  static String volume(int? volume) {
    if (volume == null) return '-';
    return NumberFormat.decimalPattern('tr_TR').format(volume);
  }
}
```

---

## 🚀 Uygulama Başlangıcı

### main.dart
```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/auth_provider.dart';
import 'providers/watchlist_provider.dart';
import 'screens/login_screen.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(create: (_) => WatchlistProvider()),
      ],
      child: MaterialApp(
        title: 'BIST AI Hisse Takip',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primarySwatch: Colors.blue,
          visualDensity: VisualDensity.adaptivePlatformDensity,
          fontFamily: 'Roboto',
        ),
        home: Consumer<AuthProvider>(
          builder: (context, auth, _) {
            return auth.isAuthenticated ? HomeScreen() : LoginScreen();
          },
        ),
      ),
    );
  }
}
```

---

## 📊 API Kullanım Örnekleri

### Uygulama Açılışında (Watchlist Yükleme)
```dart
Future<void> loadDashboard() async {
  // 1. Watchlist'i getir
  final watchlist = await APIService().getWatchlist();
  // Sonuç: 6 hisse

  // 2. Tüm hisseler için tahminleri getir (TEK İSTEK!)
  final symbols = watchlist.map((w) => w.symbol).toList();
  final predictions = await APIService().getBatchPredictions(symbols);
  // Sonuç: {'AEFES': Prediction(...), 'ARCLK': Prediction(...), ...}

  // 3. Tüm hisseler için analizleri getir (TEK İSTEK!)
  final analyses = await APIService().getBatchPatternAnalysis(symbols);
  // Sonuç: {'AEFES': PatternAnalysis(...), ...}

  // TOPLAM: 3 HTTP isteği ile tüm dashboard yüklendi! ⚡
}
```

### Hisse Ekleme
```dart
Future<void> addStock(String symbol) async {
  // 1. Watchlist'e ekle
  final item = await APIService().addToWatchlist(symbol: symbol);
  
  // 2. WebSocket'e subscribe ol
  WebSocketService().subscribeToStock(symbol);
  
  // 3. Data'yı yükle
  final prediction = await APIService().getUserPrediction(symbol);
  final analysis = await APIService().getPatternAnalysis(symbol);
  
  // UI'ı güncelle
  setState(() {
    watchlist.add(item);
    predictions[symbol] = prediction;
    analyses[symbol] = analysis;
  });
}
```

### Detay Sayfası Açma
```dart
Future<void> openDetail(String symbol) async {
  // Paralel olarak hem analiz hem fiyat geçmişi
  final results = await Future.wait([
    APIService().getPatternAnalysis(symbol, fast: true),
    APIService().getStockPrices(symbol, days: 60),
  ]);
  
  final analysis = results[0] as PatternAnalysis;
  final priceHistory = results[1] as List<StockPrice>;
  
  // Detay ekranına git
  Navigator.push(
    context,
    MaterialPageRoute(
      builder: (_) => StockDetailScreen(
        symbol: symbol,
        analysis: analysis,
        priceHistory: priceHistory,
      ),
    ),
  );
}
```

---

## 🔄 Pull-to-Refresh İmplementasyonu
```dart
Future<void> _onRefresh() async {
  await context.read<WatchlistProvider>().loadBatchData();
  
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(content: Text('Veriler güncellendi')),
  );
}

// Widget'ta kullanım:
RefreshIndicator(
  onRefresh: _onRefresh,
  child: ListView(...),
)
```

---

## 💾 Offline Destek (Hive)

### services/cache_service.dart
```dart
import 'package:hive/hive.dart';

class CacheService {
  static const String _watchlistBox = 'watchlist';
  static const String _predictionsBox = 'predictions';
  static const String _analysesBox = 'analyses';
  
  /// Cache'e watchlist kaydet
  Future<void> cacheWatchlist(List<WatchlistItem> items) async {
    final box = await Hive.openBox(_watchlistBox);
    await box.put('data', items.map((i) => i.toJson()).toList());
    await box.put('timestamp', DateTime.now().millisecondsSinceEpoch);
  }
  
  /// Cache'ten watchlist oku
  Future<List<WatchlistItem>?> getCachedWatchlist() async {
    final box = await Hive.openBox(_watchlistBox);
    final data = box.get('data');
    final timestamp = box.get('timestamp');
    
    // Cache 5 dakikadan eskiyse geçersiz
    if (timestamp != null && 
        DateTime.now().millisecondsSinceEpoch - timestamp > 300000) {
      return null;
    }
    
    if (data != null) {
      return (data as List).map((json) => WatchlistItem.fromJson(json)).toList();
    }
    
    return null;
  }
}
```

---

## 📱 Ekran Görüntüleri ve Açıklamalar

### Ana Ekran (HomeScreen)
```
┌─────────────────────────────┐
│ ⬅️  BIST AI Hisse Takip  🟢 │ ← AppBar (WiFi durumu)
├─────────────────────────────┤
│                             │
│  ┌───────────────────────┐  │
│  │ AEFES  Anadolu Efes   │  │ ← Stock Card
│  │ ₺14.02                │  │
│  │ ⬆️ Bekleme (%69)       │  │ ← Signal
│  │                       │  │
│  │ 1G:₺14.03 3G:₺14.04   │  │ ← Predictions
│  │ 7G:₺14.06 14G:₺14.12  │  │
│  │ 30G:₺14.22            │  │
│  │                       │  │
│  │ [Gelişmiş 7D] [Çekiç] │  │ ← Badges
│  │                       │  │
│  │         🗑️  📋 Detay   │  │ ← Actions
│  └───────────────────────┘  │
│                             │
│  ┌───────────────────────┐  │
│  │ ARCLK  Arçelik        │  │
│  │ ₺117.20               │  │
│  │ ...                   │  │
│  └───────────────────────┘  │
│                             │
└─────────────────────────────┘
              [+]              ← FAB (Hisse Ekle)
```

---

## 🎯 API Çağrı Stratejisi

### İlk Yükleme (Cold Start)
```dart
1. GET  /api/watchlist                    → 100ms
2. POST /api/batch/predictions            → 200ms  (6 sembol)
3. POST /api/batch/pattern-analysis       → 300ms  (6 sembol)
4. WebSocket connect + join_user          → 150ms
5. WebSocket subscribe to 6 stocks        → 50ms
────────────────────────────────────────────────
TOPLAM: ~800ms (çok hızlı!)
```

### Yenileme (Pull-to-Refresh)
```dart
1. POST /api/batch/predictions            → 200ms
2. POST /api/batch/pattern-analysis       → 300ms
────────────────────────────────────────────────
TOPLAM: ~500ms
```

### Detay Açma
```dart
1. GET /api/pattern-analysis/{symbol}     → 150ms  (cache-only)
2. GET /api/stock-prices/{symbol}         → 100ms
────────────────────────────────────────────────
TOPLAM: ~250ms
```

---

## 🔔 Bildirimler (Push Notifications)

### Canlı Sinyal Bildirimi
```dart
// WebSocket'ten gelen sinyal
_ws.liveSignalStream.listen((data) {
  final symbol = data['signal']?['symbol'];
  final signalType = data['signal']?['overall_signal']?['signal'];
  final confidence = data['signal']?['overall_signal']?['confidence'];
  
  // Local notification göster
  showNotification(
    title: '$symbol Yeni Sinyal',
    body: '$signalType (%${(confidence * 100).toInt()})',
    payload: symbol,
  );
});
```

---

## ✅ Kontrol Listesi

### Temel Özellikler
- [ ] Login/Logout
- [ ] Watchlist görüntüleme
- [ ] Hisse ekleme/çıkarma
- [ ] Tahminleri gösterme (1D/3D/7D/14D/30D)
- [ ] Sinyalleri gösterme
- [ ] Rozetleri gösterme
- [ ] Pull-to-refresh

### Detay Özellikleri
- [ ] Fiyat grafiği
- [ ] Pattern overlay (kırmızı vurgular)
- [ ] Formasyonlar listesi
- [ ] ML Özet (tüm horizon'lar)
- [ ] Evidence detayları
- [ ] Teknik göstergeler

### Gelişmiş Özellikler
- [ ] WebSocket real-time updates
- [ ] Canlı sinyal bildirimleri
- [ ] Offline destek (cache)
- [ ] Arama (debounced)
- [ ] Loading states
- [ ] Error handling

---

**Dokümantasyon hazır! Hangi bölüm için daha fazla detay istersin?** 🚀

