import click
try:
    from preprocess import preprocess_command as preprocess
    from engineer import engineer_command as engineer
    from model import model_command as model
except ImportError as e:
    # If the import fails (e.g., file renamed/moved), provide stubs
    # This fallback is less likely needed now, but keeps your original structure
    print(f"DEBUG: Import failed in AI.py: {e}")
    @click.command()
    def preprocess():
        click.echo("Command 'preprocess' not yet implemented.")
    @click.command()
    def engineer():
        click.echo("Command 'engineer' not yet implemented.")
    @click.command()
    def model():
        click.echo("Command 'model' not yet implemented.")

@click.group()
def AI():
    """CLI for AI modeling pipeline."""
    pass

# Register commands
AI.add_command(preprocess)
AI.add_command(engineer)
AI.add_command(model)

if __name__ == '__main__':
    AI()