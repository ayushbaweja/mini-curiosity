"""
Main entry point for training and comparing A2C models
"""
import argparse
import sys

from .record_videos import record_episodes, record_comparison
from .baseline_a2c import train_baseline_a2c
from .icm_a2c import train_a2c_with_icm
from .compare import compare_models
from .utils import test_model


def main():
    parser = argparse.ArgumentParser(
        description='Train and compare A2C with and without ICM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train both models and compare
  python -m curiosity_a2c.main --mode both --timesteps 200000
  
  # Train only baseline
  python -m curiosity_a2c.main --mode baseline --timesteps 200000
  
  # Train only ICM
  python -m curiosity_a2c.main --mode icm --timesteps 200000
  
  # Compare existing models
  python -m curiosity_a2c.main --mode compare --test-episodes 50
  
  # Test a specific model
  python -m curiosity_a2c.main --mode test --model-type baseline
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='both',
        choices=['baseline', 'icm', 'both', 'compare', 'test'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=200_000,
        help='Total training timesteps'
    )
    
    parser.add_argument(
        '--n-envs',
        type=int,
        default=4,
        help='Number of parallel environments'
    )
    
    parser.add_argument(
        '--baseline-path',
        type=str,
        default='models/baseline/a2c_frozenlake_baseline_final',
        help='Path to baseline model for comparison/testing'
    )
    
    parser.add_argument(
        '--icm-path',
        type=str,
        default='models/icm/a2c_frozenlake_icm_final',
        help='Path to ICM model for comparison/testing'
    )
    
    parser.add_argument(
        '--test-episodes',
        type=int,
        default=20,
        help='Number of episodes for testing/comparison'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['baseline', 'icm'],
        default='baseline',
        help='Model type to test (when mode=test)'
    )
        
    # ICM-specific parameters
    parser.add_argument(
        '--icm-lr',
        type=float,
        default=1e-3,
        help='Learning rate for ICM'
    )
    
    parser.add_argument(
        '--icm-beta',
        type=float,
        default=0.2,
        help='ICM beta parameter (forward loss weight)'
    )
    
    parser.add_argument(
        '--icm-eta',
        type=float,
        default=0.01,
        help='ICM eta parameter (intrinsic reward scale)'
    )
    
    # Add these arguments to the parser
    parser.add_argument(
        '--record-videos',
        action='store_true',
        help='Record videos after training/testing'
    )

    parser.add_argument(
        '--video-episodes',
        type=int,
        nargs='+',
        default=[1, 5, 10],
        help='Episode numbers to record as videos'
    )

    args = parser.parse_args()

    def _save_prefix(path_str: str) -> str:
        return path_str[:-6] if path_str.endswith('_final') else path_str

    # Execute based on mode
    if args.mode == 'baseline':
        print("\n>>> Training Baseline A2C <<<")
        baseline_save_path = _save_prefix(args.baseline_path)
        model, env = train_baseline_a2c(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=7e-4,
            ent_coef=0.01,
            save_path=baseline_save_path,
        )
        saved_baseline_path = f"{baseline_save_path}_final"
        args.baseline_path = saved_baseline_path
        print("\n>>> Testing Baseline A2C <<<")
        test_model(saved_baseline_path, n_episodes=10, model_type='baseline')
    
    elif args.mode == 'icm':
        print("\n>>> Training A2C with ICM <<<")
        icm_save_path = _save_prefix(args.icm_path)
        model, icm, env = train_a2c_with_icm(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=7e-4,
            ent_coef=0.01,
            icm_lr=args.icm_lr,
            icm_beta=args.icm_beta,
            icm_eta=args.icm_eta,
            save_path=icm_save_path,
        )
        saved_icm_path = f"{icm_save_path}_final"
        args.icm_path = saved_icm_path
        print("\n>>> Testing A2C with ICM <<<")
        test_model(saved_icm_path, n_episodes=10, model_type='icm')
    
    elif args.mode == 'both':
        print("\n>>> Training Both Models <<<")

        print("\n[1/2] Training Baseline A2C...")
        baseline_save_path = _save_prefix(args.baseline_path)
        baseline_model, baseline_env = train_baseline_a2c(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=7e-4,
            ent_coef=0.01,
            save_path=baseline_save_path,
        )
        saved_baseline_path = f"{baseline_save_path}_final"
        args.baseline_path = saved_baseline_path

        print("\n[2/2] Training A2C with ICM...")
        icm_save_path = _save_prefix(args.icm_path)
        icm_model, icm_module, icm_env = train_a2c_with_icm(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=7e-4,
            ent_coef=0.01,
            icm_lr=args.icm_lr,
            icm_beta=args.icm_beta,
            icm_eta=args.icm_eta,
            save_path=icm_save_path,
        )
        saved_icm_path = f"{icm_save_path}_final"
        args.icm_path = saved_icm_path

        print("\n>>> Comparing Models <<<")
        compare_models(
            args.baseline_path,
            args.icm_path,
            n_episodes=args.test_episodes
        )
    
    elif args.mode == 'compare':
        print("\n>>> Comparing Existing Models <<<")
        compare_models(
            args.baseline_path,
            args.icm_path,
            n_episodes=args.test_episodes
        )
    
    elif args.mode == 'test':
        if args.model_type == 'baseline':
            test_model(
                args.baseline_path, 
                n_episodes=args.test_episodes, 
                model_type='baseline'
            )
        else:
            test_model(
                args.icm_path, 
                n_episodes=args.test_episodes, 
                model_type='icm'
            )
    
    if args.record_videos:
        if args.mode == 'baseline':
            print("\n>>> Recording Baseline Videos <<<")
            record_episodes(
                model_path=args.baseline_path,
                episode_numbers=args.video_episodes,
                video_folder='videos/baseline',
                model_type='baseline'
            )
        elif args.mode == 'icm':
            print("\n>>> Recording ICM Videos <<<")
            record_episodes(
                model_path=args.icm_path,
                episode_numbers=args.video_episodes,
                video_folder='videos/icm',
                model_type='icm'
            )
        elif args.mode in ['both', 'compare']:
            print("\n>>> Recording Comparison Videos <<<")
            record_comparison(
                baseline_path=args.baseline_path,
                icm_path=args.icm_path,
                n_episodes=max(args.video_episodes),
                video_folder='videos/comparison'
            )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()