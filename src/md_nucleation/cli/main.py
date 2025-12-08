import typer
from pathlib import Path

app = typer.Typer()


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to the input configuration file"),
    output: str = typer.Option("mc_results.txt", help="Path to the output results file"),
    max_displacement: float = typer.Option(0.1, help="Maximum displacement for Monte Carlo moves")
):
    """
    Run a Monte Carlo simulation for molecular dynamics nucleation.
    
    Reads the configuration from the input file, builds the system,
    runs the Monte Carlo simulation, and saves results to the output file.
    """
    from md_nucleation.core.monte_carlo import run_monte_carlo_from_input
    
    # Verify input file exists
    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"Error: Input file '{config}' not found.", err=True)
        raise typer.Exit(code=1)
    
    typer.echo(f"Running Monte Carlo simulation with configuration: {config}")
    
    try:
        results = run_monte_carlo_from_input(
            input_file=config,
            output_file=output,
            max_displacement=max_displacement
        )
        typer.echo(f"\n✓ Simulation completed successfully!")
        typer.echo(f"Results saved to: {output}")
    except Exception as e:
        typer.echo(f"\n✗ Simulation failed: {str(e)}", err=True)
        raise typer.Exit(code=1)


def main():
    app()
