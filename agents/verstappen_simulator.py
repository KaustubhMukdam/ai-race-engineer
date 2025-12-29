"""
Verstappen Style Simulator
Compares aggressive driving style strategy vs conservative baseline
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import json

from agents.base_agent import BaseAgent
from agents.prompts.strategy_prompts import VERSTAPPEN_STYLE_ANALYSIS_PROMPT, SYSTEM_PROMPT
from config.app_config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VerstappenStyleSimulator(BaseAgent):
    """Simulates and compares Verstappen's aggressive style vs baseline"""
    
    def __init__(self):
        super().__init__(
            name="VerstappenSimulator",
            model=settings.groq_primary_model,
            temperature=0.7
        )
        self.degradation_df = None
        self.laps_df = None
    
    def load_race_data(self, degradation_csv: Path, laps_csv: Path):
        """Load race data for analysis"""
        try:
            self.degradation_df = pd.read_csv(degradation_csv)
            self.laps_df = pd.read_csv(laps_csv)
            logger.info(f"Loaded race data: {len(self.degradation_df)} stints, {len(self.laps_df)} laps")
        except Exception as e:
            logger.error(f"Error loading race data: {e}")
            raise
    
    def calculate_driving_style_metrics(self, driver: str) -> Dict[str, float]:
        """
        Calculate metrics that indicate driving style aggressiveness
        
        Aggressive indicators:
        - Lower average lap times (pushing harder)
        - Higher degradation rate (more tire stress)
        - More consistent pace (confidence)
        - Faster first laps on new tires (attacking immediately)
        """
        try:
            driver_laps = self.laps_df[self.laps_df['Driver'] == driver].copy()
            driver_deg = self.degradation_df[self.degradation_df['Driver'] == driver].copy()
            
            if driver_laps.empty or driver_deg.empty:
                return None
            
            # Metric 1: Average pace (excluding outliers)
            clean_laps = driver_laps[
                (driver_laps['LapTime_Seconds'] > 70) & 
                (driver_laps['LapTime_Seconds'] < 120)
            ]
            avg_pace = clean_laps['LapTime_Seconds'].mean()
            
            # Metric 2: Tire degradation aggressiveness
            avg_degradation = driver_deg['DegradationPerLap_Seconds'].mean()
            
            # Metric 3: Pace consistency (lower std = more consistent)
            pace_std = clean_laps['LapTime_Seconds'].std()
            
            # Metric 4: New tire attack (first 3 laps after pit)
            new_tire_pace = []
            for stint in driver_laps['Stint'].unique():
                stint_laps = driver_laps[driver_laps['Stint'] == stint]
                if len(stint_laps) >= 3:
                    first_3_avg = stint_laps.iloc[:3]['LapTime_Seconds'].mean()
                    if not pd.isna(first_3_avg) and first_3_avg < 120:
                        new_tire_pace.append(first_3_avg)
            
            avg_new_tire_pace = np.mean(new_tire_pace) if new_tire_pace else avg_pace
            
            # Metric 5: Stint length (aggressive = longer stints)
            avg_stint_length = driver_deg['TotalLaps'].mean()
            
            return {
                'driver': driver,
                'avg_pace': round(avg_pace, 3),
                'avg_degradation': round(avg_degradation, 4),
                'pace_consistency': round(pace_std, 3),
                'new_tire_attack': round(avg_new_tire_pace, 3),
                'avg_stint_length': round(avg_stint_length, 1),
                'total_stints': len(driver_deg)
            }
            
        except Exception as e:
            logger.error(f"Error calculating style metrics for {driver}: {e}")
            return None
    
    def classify_driving_style(self, metrics: Dict[str, float], comparison_metrics: Dict[str, float] = None) -> str:
        """
        Classify driver as Aggressive, Balanced, or Conservative
        Compares to another driver OR field average
        """
        if comparison_metrics:
            # Direct comparison mode (VER vs MAG)
            pace_diff = metrics['avg_pace'] - comparison_metrics['avg_pace']
            deg_diff = metrics['avg_degradation'] - comparison_metrics['avg_degradation']
            
            # Aggressive = faster pace AND higher degradation
            # Conservative = slower pace AND lower degradation
            
            if pace_diff < -0.5 and deg_diff > 0.02:
                return "Aggressive (Fast but hard on tires)"
            elif pace_diff < -0.5 and deg_diff <= 0.02:
                return "Aggressive (Fast and tire-friendly)"
            elif pace_diff > 0.5 and deg_diff < -0.02:
                return "Conservative (Slow but tire-saving)"
            elif pace_diff > 0.5 and deg_diff >= -0.02:
                return "Conservative (Slow pace)"
            else:
                return "Balanced"
        else:
            # Field average comparison mode
            if self.degradation_df is not None:
                avg_field_degradation = self.degradation_df['DegradationPerLap_Seconds'].mean()
                avg_field_pace = self.laps_df['LapTime_Seconds'].mean()
                
                pace_vs_field = metrics['avg_pace'] - avg_field_pace
                deg_vs_field = metrics['avg_degradation'] - avg_field_degradation
                
                if pace_vs_field < -0.5 and deg_vs_field > 0.01:
                    return "Aggressive"
                elif pace_vs_field > 0.5 and deg_vs_field < -0.01:
                    return "Conservative"
                else:
                    return "Balanced"
            
            return "Unknown"

    def compare_verstappen_vs_baseline(
        self,
        verstappen_driver: str = "VER",
        baseline_driver: str = None
    ) -> Dict[str, Any]:
        """
        Compare Verstappen's metrics vs a baseline driver or field average
        
        Args:
            verstappen_driver: Driver code for Verstappen (default: VER)
            baseline_driver: Baseline driver code (None = use most conservative driver)
        
        Returns:
            Comparison analysis
        """
        try:
            # Get Verstappen's metrics
            ver_metrics = self.calculate_driving_style_metrics(verstappen_driver)
            
            if ver_metrics is None:
                return {
                    'status': 'error',
                    'message': f"No data found for {verstappen_driver}"
                }
            
            # Find baseline driver (most conservative)
            if baseline_driver is None:
                all_drivers = self.laps_df['Driver'].unique()
                driver_metrics = []
                
                for driver in all_drivers:
                    metrics = self.calculate_driving_style_metrics(driver)
                    if metrics:
                        driver_metrics.append(metrics)
                
                # Sort by degradation (lowest = most conservative)
                sorted_metrics = sorted(driver_metrics, key=lambda x: x['avg_degradation'])
                
                # Pick most conservative that isn't Verstappen
                for m in sorted_metrics:
                    if m['driver'] != verstappen_driver:
                        baseline_metrics = m
                        baseline_driver = m['driver']
                        break
            else:
                baseline_metrics = self.calculate_driving_style_metrics(baseline_driver)
            
            if baseline_metrics is None:
                return {
                    'status': 'error',
                    'message': f"No data found for baseline driver"
                }
            
            # Calculate differences
            pace_diff = ver_metrics['avg_pace'] - baseline_metrics['avg_pace']
            deg_diff = ver_metrics['avg_degradation'] - baseline_metrics['avg_degradation']
            consistency_diff = ver_metrics['pace_consistency'] - baseline_metrics['pace_consistency']
            stint_diff = ver_metrics['avg_stint_length'] - baseline_metrics['avg_stint_length']
            
            # Strategic implications
            strategy_impact = self._calculate_strategy_impact(
                ver_metrics, baseline_metrics, pace_diff, deg_diff
            )
            
            # Around line 185, update the return statement:
            return {
                'status': 'success',
                'verstappen': {
                    **ver_metrics,
                    'style': self.classify_driving_style(ver_metrics, baseline_metrics)  # ADD baseline_metrics
                },
                'baseline': {
                    **baseline_metrics,
                    'style': self.classify_driving_style(baseline_metrics, ver_metrics)  # ADD ver_metrics
                },
                'differences': {
                    'pace_diff': round(pace_diff, 3),
                    'degradation_diff': round(deg_diff, 4),
                    'consistency_diff': round(consistency_diff, 3),
                    'stint_length_diff': round(stint_diff, 1)
                },
                'strategy_impact': strategy_impact
            }

            
        except Exception as e:
            logger.error(f"Error comparing styles: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _calculate_strategy_impact(
        self,
        ver_metrics: Dict,
        baseline_metrics: Dict,
        pace_diff: float,
        deg_diff: float
    ) -> Dict[str, Any]:
        """Calculate strategic implications of different styles"""
        
        # Pit stop timing impact
        if deg_diff > 0:
            pit_laps_earlier = int(deg_diff * 100)  # Rough estimate
            impact = f"Verstappen needs to pit ~{pit_laps_earlier} laps earlier"
        else:
            pit_laps_later = int(abs(deg_diff) * 100)
            impact = f"Verstappen can extend stint ~{pit_laps_later} laps longer"
        
        # Race time impact (over 50 laps)
        race_laps = 50
        time_gained_pace = pace_diff * race_laps
        time_lost_pits = 0
        
        if deg_diff > 0:
            # More aggressive = more pit stops potentially
            extra_pit_stops = 0.5  # Estimate
            time_lost_pits = extra_pit_stops * 22  # 22 second pit loss
        
        net_advantage = time_gained_pace - time_lost_pits
        
        return {
            'pit_timing_impact': impact,
            'time_gained_from_pace': round(time_gained_pace, 2),
            'time_lost_from_extra_pits': round(time_lost_pits, 2),
            'net_race_advantage': round(net_advantage, 2),
            'verdict': "Aggressive style wins" if net_advantage < 0 else "Conservative style wins"
        }
    
    def generate_llm_analysis(
        self,
        comparison: Dict[str, Any]
    ) -> str:
        """Generate natural language analysis using LLM"""
        try:
            # Format comparison data for prompt
            verstappen_data = json.dumps(comparison['verstappen'], indent=2)
            baseline_data = json.dumps(comparison['baseline'], indent=2)
            
            user_prompt = VERSTAPPEN_STYLE_ANALYSIS_PROMPT.format(
                verstappen_data=verstappen_data,
                baseline_data=baseline_data
            )
            
            response = self.call_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating LLM analysis: {e}")
            return f"Error: {str(e)}"
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process verstappen simulator requests"""
        action = input_data.get('action', 'compare')
        
        if action == 'compare':
            return self.compare_verstappen_vs_baseline(
                **input_data.get('params', {})
            )
        elif action == 'analyze':
            comparison = input_data.get('comparison')
            analysis = self.generate_llm_analysis(comparison)
            return {'analysis': analysis}
        else:
            return {'status': 'error', 'message': f'Unknown action: {action}'}


def main():
    """Test Verstappen simulator"""
    simulator = VerstappenStyleSimulator()
    
    # Load 2024 Abu Dhabi GP data
    from config.app_config import settings
    
    session_key = "2024_Abu_Dhabi_Grand_Prix_Race"
    processed_path = settings.processed_data_dir / f"{session_key}_processed"
    
    deg_file = processed_path / "tire_degradation_analysis.csv"
    laps_file = processed_path / "processed_laps.csv"
    
    simulator.load_race_data(deg_file, laps_file)
    
    print("\n" + "="*80)
    print("VERSTAPPEN STYLE SIMULATOR - 2024 ABU DHABI GP")
    print("="*80)
    
    # Test 1: Calculate Verstappen's metrics
    print("\n🏎️ TEST 1: Verstappen Driving Style Metrics")
    print("-"*80)
    ver_metrics = simulator.calculate_driving_style_metrics("VER")
    print(json.dumps(ver_metrics, indent=2))
    print(f"\nStyle Classification: {simulator.classify_driving_style(ver_metrics)}")
    
    # Test 2: Compare vs most conservative driver
    print("\n\n⚖️ TEST 2: Verstappen vs Conservative Baseline")
    print("-"*80)
    comparison = simulator.compare_verstappen_vs_baseline()
    
    if comparison['status'] == 'success':
        print(f"\nVerstappen: {comparison['verstappen']['driver']} ({comparison['verstappen']['style']})")
        print(f"Baseline: {comparison['baseline']['driver']} ({comparison['baseline']['style']})")
        
        print("\n📊 Key Differences:")
        diffs = comparison['differences']
        print(f"  Pace: {diffs['pace_diff']:.3f}s faster" if diffs['pace_diff'] < 0 else f"  Pace: {abs(diffs['pace_diff']):.3f}s slower")
        print(f"  Degradation: {diffs['degradation_diff']:.4f}s/lap {'higher' if diffs['degradation_diff'] > 0 else 'lower'}")
        print(f"  Stint length: {diffs['stint_length_diff']:.1f} laps {'longer' if diffs['stint_length_diff'] > 0 else 'shorter'}")
        
        print("\n🎯 Strategy Impact:")
        impact = comparison['strategy_impact']
        print(f"  {impact['pit_timing_impact']}")
        print(f"  Net advantage over 50 laps: {impact['net_race_advantage']:.2f}s")
        print(f"  Verdict: {impact['verdict']}")
        
        # Test 3: Generate AI analysis
        print("\n\n🤖 TEST 3: AI Race Engineer Analysis")
        print("-"*80)
        analysis = simulator.generate_llm_analysis(comparison)
        print(analysis)
    else:
        print(f"Error: {comparison['message']}")
    
    print("\n" + "="*80)
    print("✅ Verstappen Simulator test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
