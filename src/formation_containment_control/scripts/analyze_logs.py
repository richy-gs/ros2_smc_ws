#!/usr/bin/env python3
"""
Formation Control Data Analysis and Visualization

Post-processing script for analyzing logged formation control data.
Provides interactive visualizations for:
- Agent states (position, velocity, acceleration)
- Formation errors
- Adaptive gain evolution
- 3D trajectories

Usage:
    python analyze_logs.py [run_directory]
    
    If run_directory is not specified, lists available runs and prompts for selection.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd

# Import data loader utilities
try:
    from data_logger_utils import load_run_data, validate_data, list_available_runs
except ImportError:
    print("Error: data_logger_utils.py not found in the same directory.")
    sys.exit(1)


class FormationDataAnalyzer:
    """
    Analyzer for formation control logged data.
    
    Provides interactive visualizations for agents, errors, and adaptive gains.
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize analyzer with run directory.
        
        Args:
            run_dir: Path to run directory containing logged data
        """
        self.run_dir = Path(run_dir)
        self.data = None
        self.metadata = None
        
        # Load data
        self._load_data()
        
        # Extract agent information
        self._extract_agent_info()
    
    def _load_data(self):
        """Load and validate data from run directory."""
        print(f"Loading data from: {self.run_dir}")
        
        try:
            self.data = load_run_data(self.run_dir)
            self.metadata = self.data.get('metadata', {})
            
            # Validate data
            is_valid, issues = validate_data(self.data)
            if not is_valid:
                print("Warning: Data validation issues found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("Data loaded successfully!")
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def _print_summary(self):
        """Print data summary."""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        if self.metadata:
            print(f"Formation Type: {self.metadata.get('formation_type', 'N/A')}")
            print(f"Control Mode: {self.metadata.get('control_mode', 'N/A')}")
            print(f"Followers: {self.metadata.get('n_followers', 'N/A')}")
            print(f"Leaders: {self.metadata.get('n_leaders', 'N/A')}")
            print(f"Control Rate: {self.metadata.get('control_rate', 'N/A')} Hz")
            print(f"Log Rate: {self.metadata.get('log_rate', 'N/A')} Hz")
        
        print("\nData Samples:")
        for key in ['states', 'velocities', 'accelerations', 'errors', 'adaptive_gains']:
            if self.data.get(key) is not None:
                print(f"  {key}: {len(self.data[key])} samples")
        
        if self.data.get('states') is not None:
            df = self.data['states']
            t_start = df['timestamp'].min()
            t_end = df['timestamp'].max()
            duration = t_end - t_start
            print(f"\nDuration: {duration:.2f} seconds")
        
        print("="*60 + "\n")
    
    def _extract_agent_info(self):
        """Extract agent IDs and types from data."""
        if self.data.get('states') is not None:
            df = self.data['states']
            self.all_agents = df[['agent_id', 'agent_type']].drop_duplicates().sort_values(['agent_type', 'agent_id'])
            self.followers = self.all_agents[self.all_agents['agent_type'] == 'follower']
            self.leaders = self.all_agents[self.all_agents['agent_type'] == 'leader']
            
            self.n_followers = len(self.followers)
            self.n_leaders = len(self.leaders)
        else:
            self.all_agents = pd.DataFrame()
            self.followers = pd.DataFrame()
            self.leaders = pd.DataFrame()
            self.n_followers = 0
            self.n_leaders = 0
    
    def plot_agent_states(self, agent_id: int, agent_type: str):
        """
        Plot position, velocity, and acceleration for a single agent.
        
        Args:
            agent_id: Agent ID
            agent_type: 'leader' or 'follower'
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'{agent_type.capitalize()} {agent_id} - State Evolution', fontsize=16)
        
        # Plot position
        if self.data.get('states') is not None:
            df = self.data['states']
            agent_data = df[(df['agent_id'] == agent_id) & (df['agent_type'] == agent_type)]
            
            if not agent_data.empty:
                t = agent_data['timestamp'].values
                axes[0].plot(t, agent_data['x'].values, label='x', linewidth=2)
                axes[0].plot(t, agent_data['y'].values, label='y', linewidth=2)
                axes[0].plot(t, agent_data['z'].values, label='z', linewidth=2)
                axes[0].plot(t, agent_data['yaw'].values, label='yaw', linewidth=2)
                axes[0].set_ylabel('Position [m, rad]')
                axes[0].legend(loc='upper right')
                axes[0].grid(True, alpha=0.3)
                axes[0].set_title('Position')
        
        # Plot velocity
        if self.data.get('velocities') is not None:
            df = self.data['velocities']
            agent_data = df[(df['agent_id'] == agent_id) & (df['agent_type'] == agent_type)]
            
            if not agent_data.empty:
                t = agent_data['timestamp'].values
                axes[1].plot(t, agent_data['vx'].values, label='vx', linewidth=2)
                axes[1].plot(t, agent_data['vy'].values, label='vy', linewidth=2)
                axes[1].plot(t, agent_data['vz'].values, label='vz', linewidth=2)
                axes[1].plot(t, agent_data['vyaw'].values, label='vyaw', linewidth=2)
                axes[1].set_ylabel('Velocity [m/s, rad/s]')
                axes[1].legend(loc='upper right')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_title('Velocity')
        
        # Plot acceleration
        if self.data.get('accelerations') is not None:
            df = self.data['accelerations']
            agent_data = df[(df['agent_id'] == agent_id) & (df['agent_type'] == agent_type)]
            
            if not agent_data.empty:
                t = agent_data['timestamp'].values
                axes[2].plot(t, agent_data['ax'].values, label='ax', linewidth=2)
                axes[2].plot(t, agent_data['ay'].values, label='ay', linewidth=2)
                axes[2].plot(t, agent_data['az'].values, label='az', linewidth=2)
                axes[2].plot(t, agent_data['omega'].values, label='omega', linewidth=2)
                axes[2].set_ylabel('Acceleration [m/s², rad/s²]')
                axes[2].set_xlabel('Time [s]')
                axes[2].legend(loc='upper right')
                axes[2].grid(True, alpha=0.3)
                axes[2].set_title('Acceleration (Control Input)')
        
        plt.tight_layout()
        return fig
    
    def plot_all_agents_interactive(self):
        """Create interactive plot window for all agents with selector buttons."""
        self._create_agent_selector_window(self.all_agents, "All Agents")
    
    def plot_leaders_interactive(self):
        """Create interactive plot window for leaders with selector buttons."""
        self._create_agent_selector_window(self.leaders, "Leaders")
    
    def plot_followers_interactive(self):
        """Create interactive plot window for followers with selector buttons."""
        self._create_agent_selector_window(self.followers, "Followers")
    
    def _create_agent_selector_window(self, agents_df: pd.DataFrame, title: str):
        """
        Create interactive window with agent selector buttons.
        
        Args:
            agents_df: DataFrame with agent_id and agent_type columns
            title: Window title
        """
        if agents_df.empty:
            print(f"No agents available for {title}")
            return
        
        # Create main figure with button area
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # Create subplot for buttons (top area)
        button_ax = plt.subplot2grid((10, 1), (0, 0), rowspan=1)
        button_ax.axis('off')
        
        # Create subplot for plots (rest of the area)
        plot_container = plt.subplot2grid((10, 1), (1, 0), rowspan=9)
        plot_container.axis('off')
        
        # State to track current selection
        state = {'current_agent_id': None, 'current_agent_type': None, 'axes': []}
        
        def update_plot(agent_id, agent_type):
            """Update plot for selected agent."""
            # Clear previous plots
            for ax in state['axes']:
                ax.remove()
            state['axes'] = []
            
            # Create new subplots
            ax1 = plt.subplot2grid((10, 1), (1, 0), rowspan=3, fig=fig)
            ax2 = plt.subplot2grid((10, 1), (4, 0), rowspan=3, fig=fig)
            ax3 = plt.subplot2grid((10, 1), (7, 0), rowspan=3, fig=fig)
            state['axes'] = [ax1, ax2, ax3]
            
            # Plot position
            if self.data.get('states') is not None:
                df = self.data['states']
                agent_data = df[(df['agent_id'] == agent_id) & (df['agent_type'] == agent_type)]
                
                if not agent_data.empty:
                    t = agent_data['timestamp'].values
                    ax1.plot(t, agent_data['x'].values, label='x', linewidth=2)
                    ax1.plot(t, agent_data['y'].values, label='y', linewidth=2)
                    ax1.plot(t, agent_data['z'].values, label='z', linewidth=2)
                    ax1.plot(t, agent_data['yaw'].values, label='yaw', linewidth=2)
                    ax1.set_ylabel('Position [m, rad]')
                    ax1.legend(loc='upper right')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_title(f'{agent_type.capitalize()} {agent_id} - Position')
            
            # Plot velocity
            if self.data.get('velocities') is not None:
                df = self.data['velocities']
                agent_data = df[(df['agent_id'] == agent_id) & (df['agent_type'] == agent_type)]
                
                if not agent_data.empty:
                    t = agent_data['timestamp'].values
                    ax2.plot(t, agent_data['vx'].values, label='vx', linewidth=2)
                    ax2.plot(t, agent_data['vy'].values, label='vy', linewidth=2)
                    ax2.plot(t, agent_data['vz'].values, label='vz', linewidth=2)
                    ax2.plot(t, agent_data['vyaw'].values, label='vyaw', linewidth=2)
                    ax2.set_ylabel('Velocity [m/s, rad/s]')
                    ax2.legend(loc='upper right')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title('Velocity')
            
            # Plot acceleration
            if self.data.get('accelerations') is not None:
                df = self.data['accelerations']
                agent_data = df[(df['agent_id'] == agent_id) & (df['agent_type'] == agent_type)]
                
                if not agent_data.empty:
                    t = agent_data['timestamp'].values
                    ax3.plot(t, agent_data['ax'].values, label='ax', linewidth=2)
                    ax3.plot(t, agent_data['ay'].values, label='ay', linewidth=2)
                    ax3.plot(t, agent_data['az'].values, label='az', linewidth=2)
                    ax3.plot(t, agent_data['omega'].values, label='omega', linewidth=2)
                    ax3.set_ylabel('Acceleration [m/s², rad/s²]')
                    ax3.set_xlabel('Time [s]')
                    ax3.legend(loc='upper right')
                    ax3.grid(True, alpha=0.3)
                    ax3.set_title('Acceleration (Control Input)')
            
            state['current_agent_id'] = agent_id
            state['current_agent_type'] = agent_type
            
            plt.draw()
        
        # Create buttons for each agent
        n_agents = len(agents_df)
        button_width = 0.08
        button_height = 0.04
        button_spacing = 0.01
        buttons = []
        
        for idx, (_, row) in enumerate(agents_df.iterrows()):
            agent_id = int(row['agent_id'])
            agent_type = row['agent_type']
            
            # Position button
            x_pos = 0.1 + idx * (button_width + button_spacing)
            button_ax_pos = plt.axes([x_pos, 0.92, button_width, button_height])
            
            label = f"{agent_type[0].upper()}{agent_id}"
            btn = Button(button_ax_pos, label)
            
            # Create closure to capture current agent_id and agent_type
            def make_callback(aid, atype):
                return lambda event: update_plot(aid, atype)
            
            btn.on_clicked(make_callback(agent_id, agent_type))
            buttons.append(btn)
        
        # Plot first agent by default
        first_agent = agents_df.iloc[0]
        update_plot(int(first_agent['agent_id']), first_agent['agent_type'])
        
        # Store buttons to prevent garbage collection
        fig.buttons = buttons
        
        plt.show()
    
    def plot_formation_errors(self):
        """Plot formation errors for all agents over time."""
        if self.data.get('errors') is None or self.data['errors'].empty:
            print("No error data available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Formation Errors', fontsize=16)
        
        df = self.data['errors']
        
        # Plot leader errors
        leader_data = df[df['agent_type'] == 'leader']
        if not leader_data.empty:
            for agent_id in leader_data['agent_id'].unique():
                agent_data = leader_data[leader_data['agent_id'] == agent_id]
                t = agent_data['timestamp'].values
                error_norm = agent_data['error_norm'].values
                axes[0].plot(t, error_norm, label=f'Leader {int(agent_id)}', linewidth=2)
            
            axes[0].set_ylabel('Error Norm [m]')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('Leader Formation Errors')
        
        # Plot follower errors
        follower_data = df[df['agent_type'] == 'follower']
        if not follower_data.empty:
            for agent_id in follower_data['agent_id'].unique():
                agent_data = follower_data[follower_data['agent_id'] == agent_id]
                t = agent_data['timestamp'].values
                error_norm = agent_data['error_norm'].values
                axes[1].plot(t, error_norm, label=f'Follower {int(agent_id)}', linewidth=2)
            
            axes[1].set_ylabel('Error Norm [m]')
            axes[1].set_xlabel('Time [s]')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Follower Containment Errors')
        
        plt.tight_layout()
        return fig
    
    def plot_adaptive_gains(self):
        """Plot adaptive gain K_c evolution for all agents."""
        if self.data.get('adaptive_gains') is None or self.data['adaptive_gains'].empty:
            print("No adaptive gain data available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Adaptive Gain K_c Evolution', fontsize=16)
        
        df = self.data['adaptive_gains']
        
        # Plot leader gains (average of x, y, z, yaw)
        leader_data = df[df['agent_type'] == 'leader']
        if not leader_data.empty:
            for agent_id in leader_data['agent_id'].unique():
                agent_data = leader_data[leader_data['agent_id'] == agent_id]
                t = agent_data['timestamp'].values
                # Average K_c across all dimensions
                K_c_avg = agent_data[['Kc_x', 'Kc_y', 'Kc_z', 'Kc_yaw']].mean(axis=1).values
                axes[0].plot(t, K_c_avg, label=f'Leader {int(agent_id)}', linewidth=2)
            
            axes[0].set_ylabel('K_c (average)')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('Leader Adaptive Gains')
        
        # Plot follower gains
        follower_data = df[df['agent_type'] == 'follower']
        if not follower_data.empty:
            for agent_id in follower_data['agent_id'].unique():
                agent_data = follower_data[follower_data['agent_id'] == agent_id]
                t = agent_data['timestamp'].values
                # Average K_c across all dimensions
                K_c_avg = agent_data[['Kc_x', 'Kc_y', 'Kc_z', 'Kc_yaw']].mean(axis=1).values
                axes[1].plot(t, K_c_avg, label=f'Follower {int(agent_id)}', linewidth=2)
            
            axes[1].set_ylabel('K_c (average)')
            axes[1].set_xlabel('Time [s]')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Follower Adaptive Gains')
        
        plt.tight_layout()
        return fig
    
    def plot_trajectories_3d(self):
        """Plot 3D trajectories of all agents."""
        if self.data.get('states') is None or self.data['states'].empty:
            print("No state data available")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        df = self.data['states']
        
        # Plot leader trajectories
        leader_data = df[df['agent_type'] == 'leader']
        for agent_id in leader_data['agent_id'].unique():
            agent_data = leader_data[leader_data['agent_id'] == agent_id]
            x = agent_data['x'].values
            y = agent_data['y'].values
            z = agent_data['z'].values
            ax.plot(x, y, z, label=f'Leader {int(agent_id)}', linewidth=2, alpha=0.8)
            # Mark start and end
            ax.scatter(x[0], y[0], z[0], marker='o', s=100, alpha=0.8)
            ax.scatter(x[-1], y[-1], z[-1], marker='*', s=200, alpha=0.8)
        
        # Plot follower trajectories
        follower_data = df[df['agent_type'] == 'follower']
        for agent_id in follower_data['agent_id'].unique():
            agent_data = follower_data[follower_data['agent_id'] == agent_id]
            x = agent_data['x'].values
            y = agent_data['y'].values
            z = agent_data['z'].values
            ax.plot(x, y, z, label=f'Follower {int(agent_id)}', linewidth=2, 
                   linestyle='--', alpha=0.6)
            # Mark start and end
            ax.scatter(x[0], y[0], z[0], marker='o', s=100, alpha=0.6)
            ax.scatter(x[-1], y[-1], z[-1], marker='*', s=200, alpha=0.6)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Trajectories (o=start, *=end)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def generate_all_plots(self, save_dir: Path = None):
        """
        Generate all plots and optionally save them.
        
        Args:
            save_dir: Directory to save plots (if None, only displays)
        """
        print("\nGenerating plots...")
        
        # Formation errors
        print("  - Formation errors")
        fig = self.plot_formation_errors()
        if fig and save_dir:
            fig.savefig(save_dir / 'formation_errors.png', dpi=150, bbox_inches='tight')
        
        # Adaptive gains
        print("  - Adaptive gains")
        fig = self.plot_adaptive_gains()
        if fig and save_dir:
            fig.savefig(save_dir / 'adaptive_gains.png', dpi=150, bbox_inches='tight')
        
        # 3D trajectories
        print("  - 3D trajectories")
        fig = self.plot_trajectories_3d()
        if fig and save_dir:
            fig.savefig(save_dir / 'trajectories_3d.png', dpi=150, bbox_inches='tight')
        
        # Individual agent plots
        print("  - Individual agent states")
        if save_dir:
            agent_dir = save_dir / 'agents'
            agent_dir.mkdir(exist_ok=True)
            
            for _, row in self.all_agents.iterrows():
                agent_id = int(row['agent_id'])
                agent_type = row['agent_type']
                fig = self.plot_agent_states(agent_id, agent_type)
                fig.savefig(agent_dir / f'{agent_type}_{agent_id}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        if save_dir:
            print(f"\nPlots saved to: {save_dir}")
        
        print("Done!")


def select_run_interactive(base_dir: Path = Path("logs")) -> Path:
    """
    Interactively select a run from available runs.
    
    Args:
        base_dir: Base directory containing run folders
        
    Returns:
        Path to selected run directory
    """
    runs = list_available_runs(base_dir)
    
    if not runs:
        print(f"No runs found in {base_dir}")
        sys.exit(1)
    
    print("\nAvailable runs:")
    print("-" * 40)
    for i, (run_name, run_path) in enumerate(runs, 1):
        print(f"{i}. {run_name}")
    print("-" * 40)
    
    while True:
        try:
            choice = input(f"\nSelect run (1-{len(runs)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                sys.exit(0)
            
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                return runs[idx][1]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(runs)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(
        description='Analyze formation control logged data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_logs.py                    # Interactive run selection
  python analyze_logs.py logs/run_001       # Analyze specific run
  python analyze_logs.py logs/run_001 --save plots  # Save plots to directory
        """
    )
    
    parser.add_argument('run_dir', nargs='?', help='Path to run directory')
    parser.add_argument('--save', metavar='DIR', help='Save plots to directory')
    parser.add_argument('--base-dir', default='logs', help='Base logs directory (default: logs)')
    
    args = parser.parse_args()
    
    # Select run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
    else:
        run_dir = select_run_interactive(Path(args.base_dir))
    
    # Create analyzer
    analyzer = FormationDataAnalyzer(run_dir)
    
    # Generate plots
    save_dir = None
    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving plots to: {save_dir}")
    
    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("ANALYSIS MENU")
        print("="*60)
        print("1. View All Agents (interactive)")
        print("2. View Leaders (interactive)")
        print("3. View Followers (interactive)")
        print("4. View Formation Errors")
        print("5. View Adaptive Gains")
        print("6. View 3D Trajectories")
        print("7. Generate All Plots")
        if not args.save:
            print("8. Save All Plots")
        print("q. Quit")
        print("="*60)
        
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == '1':
            analyzer.plot_all_agents_interactive()
        elif choice == '2':
            analyzer.plot_leaders_interactive()
        elif choice == '3':
            analyzer.plot_followers_interactive()
        elif choice == '4':
            analyzer.plot_formation_errors()
            plt.show()
        elif choice == '5':
            analyzer.plot_adaptive_gains()
            plt.show()
        elif choice == '6':
            analyzer.plot_trajectories_3d()
            plt.show()
        elif choice == '7':
            analyzer.generate_all_plots(save_dir)
            if not save_dir:
                plt.show()
        elif choice == '8' and not args.save:
            save_path = input("Enter directory to save plots: ").strip()
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                analyzer.generate_all_plots(save_dir)
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()

