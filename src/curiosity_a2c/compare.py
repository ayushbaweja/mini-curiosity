"""
Utilities for comparing baseline A2C and A2C + ICM
"""
import numpy as np
from scipy import stats

from .utils import test_model


def compare_models(baseline_path, icm_path, n_episodes=20):
    """
    Compare baseline A2C and ICM-enhanced A2C
    
    Args:
        baseline_path: Path to baseline model (without .zip)
        icm_path: Path to ICM model (without .zip)
        n_episodes: Number of episodes for comparison
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*70)
    print("COMPARING BASELINE A2C vs A2C + ICM")
    print("="*70)
    
    print("\n[1/2] Testing Baseline A2C...")
    baseline_rewards, baseline_lengths = test_model(
        baseline_path, 
        n_episodes=n_episodes, 
        model_type='baseline'
    )
    
    print("\n[2/2] Testing A2C + ICM...")
    icm_rewards, icm_lengths = test_model(
        icm_path, 
        n_episodes=n_episodes, 
        model_type='icm'
    )
    
    # Comparison statistics
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<30} {'Baseline A2C':<20} {'A2C + ICM':<20}")
    print("-"*70)
    print(f"{'Mean Reward':<30} {np.mean(baseline_rewards):<20.2f} {np.mean(icm_rewards):<20.2f}")
    print(f"{'Std Reward':<30} {np.std(baseline_rewards):<20.2f} {np.std(icm_rewards):<20.2f}")
    print(f"{'Mean Steps':<30} {np.mean(baseline_lengths):<20.2f} {np.mean(icm_lengths):<20.2f}")
    print(f"{'Std Steps':<30} {np.std(baseline_lengths):<20.2f} {np.std(icm_lengths):<20.2f}")
    
    baseline_success = sum(1 for r in baseline_rewards if r > -200) / n_episodes * 100
    icm_success = sum(1 for r in icm_rewards if r > -200) / n_episodes * 100
    print(f"{'Success Rate (%)':<30} {baseline_success:<20.1f} {icm_success:<20.1f}")
    
    # Statistical comparison
    t_stat, p_value = stats.ttest_ind(baseline_rewards, icm_rewards)
    print(f"\n{'Statistical Test (t-test)':<30}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        winner = "A2C + ICM" if np.mean(icm_rewards) > np.mean(baseline_rewards) else "Baseline A2C"
        improvement = abs(np.mean(icm_rewards) - np.mean(baseline_rewards))
        print(f"  Result: {winner} is significantly better (p < 0.05)")
        print(f"  Mean improvement: {improvement:.2f} reward points")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    
    print("="*70 + "\n")
    
    return {
        'baseline': {
            'rewards': baseline_rewards, 
            'lengths': baseline_lengths,
            'mean_reward': np.mean(baseline_rewards),
            'success_rate': baseline_success
        },
        'icm': {
            'rewards': icm_rewards, 
            'lengths': icm_lengths,
            'mean_reward': np.mean(icm_rewards),
            'success_rate': icm_success
        },
        'statistics': {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    }


if __name__ == "__main__":
    # Example usage
    results = compare_models(
        "a2c_mountaincar_baseline_final",
        "a2c_mountaincar_icm_final",
        n_episodes=20
    )
    
    print("\nComparison complete!")
    print(f"Baseline mean reward: {results['baseline']['mean_reward']:.2f}")
    print(f"ICM mean reward: {results['icm']['mean_reward']:.2f}")