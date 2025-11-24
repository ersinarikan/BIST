import logging


logger = logging.getLogger(__name__)


def calculate_pattern_agreement(patterns, ml_signal: str, ml_confidence: float) -> float:
    """Agreement score for ML Primary + Pattern Confirmation.

    Returns a score in [-0.20, +0.15].
    """
    try:
        PATTERN_ALIASES = {
            'DOJI': {'DOJI', 'DOJI_STAR'},
            'HAMMER': {'HAMMER'},
            'INVERTED_HAMMER': {'INVERTED_HAMMER'},
            'SHOOTING_STAR': {'SHOOTING_STAR'},
            'ENGULFING': {'ENGULFING', 'ENGULFING_BULLISH', 'ENGULFING_BEARISH'},
            'HARAMI': {'HARAMI', 'HARAMI_BULLISH', 'HARAMI_BEARISH'},
            'MORNING_STAR': {'MORNING_STAR', 'MORNING_DOJI_STAR'},
            'EVENING_STAR': {'EVENING_STAR', 'EVENING_DOJI_STAR'},
            'PIERCING': {'PIERCING', 'PIERCING_LINE'},
            'DARK_CLOUD': {'DARK_CLOUD', 'DARK_CLOUD_COVER'},
            'THREE_WHITE_SOLDIERS': {'THREE_WHITE_SOLDIERS', '3_WHITE_SOLDIERS'},
            'THREE_BLACK_CROWS': {'THREE_BLACK_CROWS', '3_BLACK_CROWS'},
        }

        def get_pattern_family(pattern_name: str) -> str:
            pu = (pattern_name or '').upper()
            for family, aliases in PATTERN_ALIASES.items():
                if pu in aliases or any(alias in pu for alias in aliases):
                    return family
            return pattern_name

        bullish = []
        bearish = []
        seen_families = set()
        for p in (patterns or []):
            try:
                if str(p.get('source', '')).upper() in ('ML', 'ML_PREDICTOR', 'ENH', 'ENHANCED_ML'):
                    continue
                fam = get_pattern_family(str(p.get('pattern', '')))
                if fam in seen_families:
                    continue
                seen_families.add(fam)
                sig = str(p.get('signal', '')).upper()
                conf = float(p.get('confidence', 0.5))
                if sig == 'BULLISH':
                    bullish.append(conf)
                elif sig == 'BEARISH':
                    bearish.append(conf)
            except Exception:
                continue

        agree_count = 0
        opp_count = 0
        if str(ml_signal).upper() == 'BULLISH':
            agree_count = sum(1 for c in bullish if c >= 0.5)
            opp_count = sum(1 for c in bearish if c >= 0.5)
        elif str(ml_signal).upper() == 'BEARISH':
            agree_count = sum(1 for c in bearish if c >= 0.5)
            opp_count = sum(1 for c in bullish if c >= 0.5)

        base = 0.0
        if agree_count >= 3:
            base = 0.15
        elif agree_count == 2:
            base = 0.10
        elif agree_count == 1:
            base = 0.05
        else:
            base = 0.0

        penalty = 0.0
        if opp_count >= 2:
            penalty = -0.20
        elif opp_count == 1:
            penalty = -0.10

        # Confidence modulation (softer impact when ML conf is low)
        try:
            mlc = max(0.0, min(1.0, float(ml_confidence)))
        except Exception:
            mlc = 0.6
        mod = 0.5 + 0.5 * mlc
        score = (base + penalty) * mod
        return max(-0.20, min(0.15, float(score)))
    except Exception as e:
        logger.debug(f"pattern agreement error: {e}")
        return 0.0
