// Pattern İsimleri Türkçe Çeviri
const PATTERN_TRANSLATIONS = {
    // Advanced TA Patterns
    'INVERSE_HEAD_AND_SHOULDERS': 'Ters Omuz Baş Omuz',
    'HEAD_AND_SHOULDERS': 'Omuz Baş Omuz',
    // Aliases for backend canonical names
    'INVERSE_HEAD_SHOULDERS': 'Ters Omuz Baş Omuz',
    'HEAD_SHOULDERS': 'Omuz Baş Omuz',
    'DOUBLE_BOTTOM': 'Çift Dip',
    'DOUBLE_TOP': 'Çift Tepe',
    'TRIANGLE_ASCENDING': 'Yükselen Üçgen',
    'TRIANGLE_DESCENDING': 'Alçalan Üçgen',
    'ASCENDING_TRIANGLE': 'Yükselen Üçgen',
    'DESCENDING_TRIANGLE': 'Alçalan Üçgen',
    'WEDGE_RISING': 'Yükselen Kama',
    'WEDGE_FALLING': 'Düşen Kama',
    'RISING_WEDGE': 'Yükselen Kama',
    'FALLING_WEDGE': 'Düşen Kama',
    'FLAG_BULLISH': 'Yükseliş Bayrağı',
    'FLAG_BEARISH': 'Düşüş Bayrağı',
    'BULLISH_FLAG': 'Yükseliş Bayrağı',
    'BEARISH_FLAG': 'Düşüş Bayrağı',
    'CUP_AND_HANDLE': 'Fincan ve Kulp',
    'CHANNEL_UP': 'Yükseliş Kanalı',
    'CHANNEL_DOWN': 'Düşüş Kanalı',
    'SUPPORT_LEVEL': 'Destek Seviyesi',
    'RESISTANCE_LEVEL': 'Direnç Seviyesi',
    
    // Basic Patterns
    'BREAKOUT_UP': 'Yukarı Kırılım',
    'BREAKDOWN': 'Aşağı Kırılım',
    'MA_CROSSOVER_BULLISH': 'Hareketli Ortalama Kesişimi (Yükseliş)',
    'MA_CROSSOVER_BEARISH': 'Hareketli Ortalama Kesişimi (Düşüş)',
    'RSI_OVERSOLD': 'RSI Aşırı Satış',
    'RSI_OVERBOUGHT': 'RSI Aşırı Alım',
    'MACD_BULLISH': 'MACD Yükseliş Sinyali',
    'MACD_BEARISH': 'MACD Düşüş Sinyali',
    
    // ML Predictions
    'ML_1D': '1 Günlük Tahmin',
    'ML_3D': '3 Günlük Tahmin', 
    'ML_7D': '1 Haftalık Tahmin',
    'ML_14D': '2 Haftalık Tahmin',
    'ML_30D': '1 Aylık Tahmin',
    
    // Visual YOLO Patterns
    'HAMMER': 'Çekiç',
    'DOJI': 'Doji',
    'SPINNING_TOP': 'Dönen Top',
    'SHOOTING_STAR': 'Düşen Yıldız',
    'ENGULFING_BULLISH': 'Yutucu Mum (Yükseliş)',
    'ENGULFING_BEARISH': 'Yutucu Mum (Düşüş)'
};

// Pattern ismini Türkçe'ye çevir
function translatePattern(patternName) {
    if (!patternName) return '';

    const upperName = patternName.toString().toUpperCase();

    // Dinamik ML pattern'leri (ör. ML_PREDICTOR_3D, ENHANCED_ML_14D)
    const mlPred = upperName.match(/^ML_PREDICTOR_(\d+)D$/);
    if (mlPred) {
        return `Temel ${mlPred[1]}G`;
    }
    const enhMl = upperName.match(/^ENHANCED_ML_(\d+)D$/);
    if (enhMl) {
        return `Gelişmiş ${enhMl[1]}G`;
    }
    if (upperName === 'FINGPT_SENTIMENT') {
        return 'Sezgisel';
    }

    // Statik eşleşmeler
    const translation = PATTERN_TRANSLATIONS[upperName]
        || PATTERN_TRANSLATIONS[upperName.replace(/^ML_PREDICTOR_/, 'ML_')] // Eski/alternatif anahtarlar
        || PATTERN_TRANSLATIONS[upperName.replace(/^ENHANCED_ML_/, 'ML_')];
    if (translation) return translation;

    // Fallback: underscore'ları boşluk yap ve title case
    return upperName
        .replace(/_/g, ' ')
        .toLowerCase()
        .replace(/\b\w/g, l => l.toUpperCase());
}

// Export for use in templates
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PATTERN_TRANSLATIONS, translatePattern };
}
