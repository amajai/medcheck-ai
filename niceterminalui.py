"""
NiceTerminalUI - Beautiful Terminal UI Components for Enhanced User Experience

A lightweight Python package that provides colorful, formatted terminal output 
functions for creating professional CLI applications with ease.

Enhanced with Rich library for advanced terminal features like tables, progress bars,
panels, and much more!
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich.box import ROUNDED, DOUBLE, HEAVY
import time

# Initialize Rich console
console = Console()

# ANSI color codes for beautiful terminal output (kept for backward compatibility)
class Colors:
    """ANSI color codes and text formatting constants"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner(
        title="APP TITLE", 
        subtitle="Cool Application", 
        description="Powered by AI Agents",
        subheader1="Subheader 1",
        subheader2="Subheader 2"
        ):
    """Print a beautiful banner with customizable content using Rich
    
    Args:
        title (str): Main application title
        subtitle (str): Application description/subtitle
        description (str): Additional description text
        subheader1 (str): First subheader line
        subheader2 (str): Second subheader line
    """
    banner_content = Align.center(f"""[bold magenta]{title}[/bold magenta]
[cyan]{subtitle}[/cyan]
[green]{description}[/green]

[cyan]{subheader1}[/cyan]
[green]{subheader2}[/green]""")
    
    console.print(Panel(
        Align.center(banner_content),
        box=DOUBLE,
        style="bold purple",
        padding=(1, 2)
    ))


def print_step(step_name, emoji="🔄"):
    """Print a formatted step indicator using Rich
    
    Args:
        step_name (str): Name of the current step
        emoji (str): Emoji to display with the step (default: 🔄)
    """
    step_text = f"{emoji} [bold blue]{step_name.upper()}[/bold blue]"
    console.print()
    console.print(Panel(step_text, box=ROUNDED, style="bold blue"))


def print_success(message):
    """Print success message with green checkmark using Rich
    
    Args:
        message (str): Success message to display
    """
    console.print(f"✅ [bold green]{message}[/bold green]")


def print_warning(message):
    """Print warning message with yellow warning icon using Rich
    
    Args:
        message (str): Warning message to display
    """
    console.print(f"⚠️  [bold yellow]{message}[/bold yellow]")


def print_error(message):
    """Print error message with red X icon using Rich
    
    Args:
        message (str): Error message to display
    """
    console.print(f"❌ [bold red]{message}[/bold red]")


def print_info(message):
    """Print info message with blue info icon using Rich
    
    Args:
        message (str): Info message to display
    """
    console.print(f"ℹ️  [bold cyan]{message}[/bold cyan]")


def print_result_box(title, content):
    """Print title in a box header and content as regular text below using Rich
    
    Args:
        title (str): Title to display in the box header
        content (str): Content to display below the box
    """
    console.print()
    console.print(Panel(
        content,
        title=f"[bold magenta]{title}[/bold magenta]",
        box=ROUNDED,
        padding=(1, 2)
    ))


def create_interactive_prompt(question, colors=Colors):
    """Create a beautiful interactive prompt for user input (legacy version)
    
    Args:
        question (str): Question to ask the user
        colors (Colors): Color class to use (default: Colors)
        
    Returns:
        str: Formatted prompt string ready for input()
    """
    prompt = f"\n{colors.OKCYAN}{colors.BOLD}💭 {question}{colors.ENDC}\n{colors.BOLD}👉 Your response: {colors.ENDC}"
    return prompt


def rich_prompt(question, default=None):
    """Create a Rich interactive prompt for user input
    
    Args:
        question (str): Question to ask the user
        default (str, optional): Default value if user presses enter
        
    Returns:
        str: User's response
    """
    return Prompt.ask(f"💭 [bold cyan]{question}[/bold cyan]", default=default)


def rich_confirm(question):
    """Create a Rich yes/no confirmation prompt
    
    Args:
        question (str): Yes/No question to ask the user
        
    Returns:
        bool: True for yes, False for no
    """
    return Confirm.ask(f"❓ [bold yellow]{question}[/bold yellow]")


def print_completion_message(app_name="App Name", slogan="The Best App"):
    """Print a final completion message using Rich
    
    Args:
        app_name (str): Name of the application
        slogan (str): Slogan or tagline for the application
    """
    message = f"✨ [bold green]Thank you for using {app_name} - {slogan}![/bold green] ✨"
    console.print()
    console.print(Panel(Align.center(message), box=DOUBLE, style="bold green"))
    console.print()


# New Rich-powered functions
def create_table(title, headers, rows, style="blue"):
    """Create a beautiful table using Rich
    
    Args:
        title (str): Table title
        headers (list): List of column headers
        rows (list): List of row data (each row is a list)
        style (str): Table style color
        
    Returns:
        Table: Rich Table object ready to print
    """
    table = Table(title=f"[bold {style}]{title}[/bold {style}]", box=ROUNDED)
    
    for header in headers:
        table.add_column(header, style=style, justify="left")
    
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    
    return table


def print_table(title, headers, rows, style="blue"):
    """Print a beautiful table using Rich
    
    Args:
        title (str): Table title
        headers (list): List of column headers  
        rows (list): List of row data (each row is a list)
        style (str): Table style color
    """
    table = create_table(title, headers, rows, style)
    console.print()
    console.print(table)
    console.print()


def create_progress_bar():
    """Create a Rich progress bar
    
    Returns:
        Progress: Rich Progress object
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )


def demo_progress_bar(description="Processing", total=100, delay=0.05):
    """Demonstrate a progress bar with fake progress
    
    Args:
        description (str): Description text
        total (int): Total steps
        delay (float): Delay between steps in seconds
    """
    with create_progress_bar() as progress:
        task = progress.add_task(f"[cyan]{description}...", total=total)
        
        for _ in range(total):
            time.sleep(delay)
            progress.update(task, advance=1)


def print_status_panel(title, status_items, style="green"):
    """Print a status panel with key-value pairs
    
    Args:
        title (str): Panel title
        status_items (dict): Dictionary of status items
        style (str): Panel style color
    """
    content = ""
    for key, value in status_items.items():
        content += f"[bold]{key}:[/bold] {value}\n"
    
    console.print()
    console.print(Panel(
        content.rstrip(),
        title=f"[bold {style}]{title}[/bold {style}]",
        box=ROUNDED,
        style=style
    ))


def print_tree_structure(title, structure, style="green"):
    """Print a tree-like structure
    
    Args:
        title (str): Tree title
        structure (dict): Nested dictionary representing the tree
        style (str): Tree style color
    """
    from rich.tree import Tree
    
    tree = Tree(f"[bold {style}]{title}[/bold {style}]")
    
    def add_items(node, items):
        if isinstance(items, dict):
            for key, value in items.items():
                if isinstance(value, dict):
                    branch = node.add(f"[bold]{key}[/bold]")
                    add_items(branch, value)
                else:
                    node.add(f"{key}: {value}")
        elif isinstance(items, list):
            for item in items:
                node.add(str(item))
    
    add_items(tree, structure)
    console.print()
    console.print(tree)
    console.print()


def print_alert(message, alert_type="info"):
    """Print an alert message with different styles
    
    Args:
        message (str): Alert message
        alert_type (str): Type of alert (info, warning, error, success)
    """
    styles = {
        "info": ("blue", "ℹ️"),
        "warning": ("yellow", "⚠️"),
        "error": ("red", "❌"),
        "success": ("green", "✅")
    }
    
    style, emoji = styles.get(alert_type, ("blue", "ℹ️"))
    
    console.print()
    console.print(Panel(
        f"{emoji} [bold]{message}[/bold]",
        box=HEAVY,
        style=style,
        padding=(1, 2)
    ))


# Export all public functions
__all__ = [
    'Colors',
    'console',
    'print_banner',
    'print_step', 
    'print_success',
    'print_warning',
    'print_error',
    'print_info',
    'print_result_box',
    'create_interactive_prompt',
    'rich_prompt',
    'rich_confirm',
    'print_completion_message',
    'create_table',
    'print_table',
    'create_progress_bar',
    'demo_progress_bar',
    'print_status_panel',
    'print_tree_structure',
    'print_alert'
]
