import click
from graphix.auth import authenticate
from graphix.task_manager import listen 
from graphix.gpu_manager import detect
from graphix.config import setup
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Define ASCII Art
GRAPHIX_ASCII_ART = r"""
  ________                    .__    .__            _____  .___ 
 /  _____/___________  ______ |  |__ |__|__  ___   /  _  \ |   |
/   \  __\_  __ \__  \ \____ \|  |  \|  \  \/  /  /  /_\  \|   |
\    \_\  \  | \// __ \|  |_> >   Y  \  |>    <  /    |    \   |
 \______  /__|  (____  /   __/|___|  /__/__/\_ \ \____|__  /___|
        \/           \/|__|        \/         \/         \/     
"""

@click.group()
def cli():
    """
======================
GRAPHIX CLI
======================
The Graphix CLI is a command-line interface that connects users with the Graphix platform. It allows users to lend their GPU for computational tasks.
By running the CLI, users can log in to the system, perform initial setup, check for available GPUs, and listen for incoming tasks. 
The CLI provides a convenient way for users to interact with the Graphix platform and contribute their GPU resources.
    """
    pass

@cli.command()
def info():
    """Graphix Info."""
    click.echo(Fore.BLUE + Style.BRIGHT + GRAPHIX_ASCII_ART + Style.RESET_ALL)
    click.echo(Fore.YELLOW + "Welcome to Graphix!")
    click.echo(Fore.CYAN+"The Graphix CLI is a command-line interface that connects users with the Graphix platform.")
    # click.echo(Style.RESET_ALL)

@cli.command()
def login():
    """Log in to the system using credentials."""
    authenticate.login()

@cli.command()
def initialsetup():
    """initial setup"""
    setup.setup_initial_config()

@cli.command()
def detech_gpus():
    """check for cpus"""
    detect.detech_gpus()

@cli.command()
def listentask():
    """Listen to incoming tasks"""
    listen.listen()

def main():
    click.echo(Fore.BLUE + Style.BRIGHT + GRAPHIX_ASCII_ART + Style.RESET_ALL)  
    cli()

if __name__ == "__main__":
    main()