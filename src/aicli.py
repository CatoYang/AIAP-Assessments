import click

# 1. Import the commands from their new files
from commands.data_loader import load_clean
from commands.data_validation import validate

@click.group()
def aicli():
    """Top-level CLI group."""
    pass

# 2. Attach the imported commands to the group
aicli.add_command(load_clean)
aicli.add_command(validate)

if __name__ == "__main__":
    aicli()