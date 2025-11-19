import typer

app = typer.Typer()


@app.command()
def run(config: str):
    typer.echo(f"Running simulation with {config}")


def main():
    app()
