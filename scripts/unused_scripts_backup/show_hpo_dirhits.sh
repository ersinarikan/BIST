#!/usr/bin/env bash
# Quick script to show HPO DirHit results from logs

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         HPO DİRHİT SONUÇLARI                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Phase 1: logs/hpo_*.log (optuna_hpo_pilot.py)
# Phase 2: logs/hpo_phase2_features_on/hpo_*.log (optuna_hpo_pilot_features_on.py)
for log in /opt/bist-pattern/logs/hpo_*.log /opt/bist-pattern/logs/hpo_phase2_features_on/hpo_*.log; do
    if [[ -f "$log" ]]; then
        # Extract symbol and horizon from filename
        filename=$(basename "$log")
        symbol=$(echo "$filename" | sed -n 's/hpo_\([A-Z0-9]*\)_\([0-9]*d\)\.log/\1/p')
        horizon=$(echo "$filename" | sed -n 's/hpo_\([A-Z0-9]*\)_\([0-9]*d\)\.log/\2/p')
        
        # Check for all .log* files (.log, .log.1, .log.2, etc.) and use the most recent one with content
        active_log="$log"
        latest_log="$log"
        latest_mtime=0
        if [[ -f "$log" ]] && [[ -s "$log" ]]; then
            latest_mtime=$(stat -c %Y "$log" 2>/dev/null || echo 0)
            latest_log="$log"
        fi
        # Check .log.1, .log.2, .log.3, etc.
        for alt_log in "${log}".*; do
            if [[ -f "$alt_log" ]] && [[ -s "$alt_log" ]]; then
                mtime=$(stat -c %Y "$alt_log" 2>/dev/null || echo 0)
                if [[ $mtime -gt $latest_mtime ]]; then
                    latest_mtime=$mtime
                    latest_log="$alt_log"
                fi
            fi
        done
        active_log="$latest_log"
        
        # Extract best DirHit robustly: scan all and take MAX (not just last)
        best_line=$(grep -E "Best is trial" "$active_log" 2>/dev/null | grep -oP "trial\s+[0-9]+.*?value:\s*[0-9.]+" | awk '{
            # find value token
            for(i=1;i<=NF;i++){if($i ~ /^value:/){val=$(i+1)+0}}
            # find trial token
            t=0; if($2 ~ /^[0-9]+$/){t=$2}
            printf("%f %d\n", val, t)
        }' | sort -k1,1nr | head -1)
        best_dh=$(echo "$best_line" | awk '{print $1}')
        best_trial=$(echo "$best_line" | awk '{print $2}')
        
        # Extract current trial number and its last value
        current_trial=$(grep -E "Trial [0-9]+ finished" "$active_log" 2>/dev/null | tail -1 | grep -oP "Trial \K[0-9]+" | head -1)
        current_val=$(grep -E "Trial [0-9]+ finished" "$active_log" 2>/dev/null | tail -1 | grep -oP "value:\s*\K[0-9.]+" | head -1)
        # Total trials finished in this run
        trials_done=$(grep -E "Trial [0-9]+ finished" "$active_log" 2>/dev/null | wc -l | awk '{print $1}')

        # Derive numeric horizon (e.g., 14 from 14d)
        hnum=$(echo "$horizon" | tr -d 'dD')
        # Detect running process for this symbol/horizon when log is empty or has no trials yet
        # Phase 1: optuna_hpo_pilot.py
        # Phase 2: optuna_hpo_pilot_features_on.py
        proc_count=$(ps aux | grep -F "/opt/bist-pattern/scripts/optuna_hpo_pilot.py --symbols $symbol --horizon $hnum" | grep -v grep | wc -l)
        proc_count2=$(ps aux | grep -F "/opt/bist-pattern/scripts/optuna_hpo_pilot_features_on.py --symbols $symbol --horizon $hnum" | grep -v grep | wc -l)
        proc_count=$((proc_count + proc_count2))
        
        # Check if optimization is complete (check both .log and .log.1)
        if grep -q "OPTIMIZATION COMPLETE" "$active_log" 2>/dev/null || grep -q "OPTIMIZATION COMPLETE" "$log" 2>/dev/null; then
            status="✅ Complete"
        elif [[ -n "$current_trial" ]]; then
            status="Trial $current_trial"
        else
            status="Starting..."
        fi
        
        # Extract last trial
        last_trial=$(grep -E "Trial [0-9]+ finished" "$log" | tail -1 | grep -oP "Trial \K[0-9]+" | head -1)
        
        if [[ -n "$symbol" && -n "$horizon" ]]; then
            if [[ -n "$best_dh" ]]; then
                if [[ -n "$current_trial" && -n "$current_val" ]]; then
                    printf "%-8s %-6s Best: %5.2f%% (trial %s) | Last: %5.2f%% (%s) | Trials: %s\n" "$symbol" "$horizon" "$best_dh" "$best_trial" "$current_val" "$status" "$trials_done"
                else
                    printf "%-8s %-6s Best: %5.2f%% (trial %s) (%s) | Trials: %s\n" "$symbol" "$horizon" "$best_dh" "$best_trial" "$status" "$trials_done"
                fi
            else
                # No best yet; refine status with process awareness
                if [[ "$proc_count" -gt 0 ]]; then
                    if [[ "$trials_done" -gt 0 ]]; then
                        printf "%-8s %-6s ⏳ Running (trials: %s)\n" "$symbol" "$horizon" "$trials_done"
                    else
                        printf "%-8s %-6s ⏳ Running (0 trials yet)\n" "$symbol" "$horizon"
                    fi
                else
                    printf "%-8s %-6s Starting...\n" "$symbol" "$horizon"
                fi
            fi
        fi
    fi
done | sort

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "JSON kayıtları:"
ls -1 /opt/bist-pattern/results/optuna_pilot_*.json 2>/dev/null | wc -l | xargs echo "Toplam JSON:" && echo ""



